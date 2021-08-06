import os

import torch
import torch.optim as optim
import math
from torch import Tensor
from typing import List
import torch.nn as nn

from run_nerf_helpers import (
    get_embedder,
    NeRF,
)

from camera_dict import camera_dict


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(
    args, noisy_focal, noisy_poses, H, W, mode="train", device="cuda"
):
    """Instantiate NeRF's MLP model.
    """

    camera_model = None

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, 
            args.i_embed
        )
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, 
                 output_ch=output_ch, skips=skips, 
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())
    
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        model_fine = nn.DataParallel(model_fine).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn, embed_fn=embed_fn, 
        embeddirs_fn=embeddirs_fn, netchunk=args.netchunk_per_gpu * args.n_gpus)
    
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train
    }
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    
    if args.camera_model != "none":
        
        cw_init = W / 2
        ch_init = H / 2
        fx_init = W if args.run_without_colmap != "none" else noisy_focal
        fy_init = H if args.run_without_colmap != "none" else noisy_focal
        
        intrinsic_init = torch.tensor(
            [
                [fx_init, 0, cw_init, 0],
                [0, fy_init, ch_init, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        
        camera_kwargs = {
            "intrinsics": intrinsic_init,
            "extrinsics": noisy_poses,
            "args": args,
            "H": H,
            "W": W,
        }
        
        with torch.no_grad():
            camera_model = camera_dict[args.camera_model](**camera_kwargs)
            camera_model = camera_model.cuda()
        
        grad_vars += list(camera_model.parameters())

    # Create optimizer
    if args.use_custom_optim:
        optimizer = CustomAdamOptimizer(
            params=grad_vars, lr=args.lrate, betas=(0.9, 0.999),
            weight_decay=args.non_linear_weight_decay, H=H, W=W, args=args
        )
    else:
        optimizer = torch.optim.Adam(
            params=grad_vars, lr=args.lrate, betas=(0.9, 0.999)
        )
        print("YES?")
    ##########################

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f) for f in sorted(
                os.listdir(os.path.join(basedir, expname))
            ) if 'tar' in f
        ]

    print('Found ckpts', ckpts)
    
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if not model_fine is None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        if not camera_model is None:
            camera_model.load_state_dict(ckpt["camera_model"])


    ##########################

    return (
        render_kwargs_train, 
        render_kwargs_test, 
        start, 
        grad_vars, 
        optimizer, 
        camera_model
    )


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def f_custom_adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         H,
         W,
         args,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    decay_idx_from = len(params)
    if args.camera_model != "none": 
        # ray_o parameters
        decay_idx_from -= "rayo" in args.camera_model
        # ray_d parameters
        decay_idx_from -= "rayd" in args.camera_model
        # distortion parameters
        decay_idx_from -= "dist" in args.camera_model
    
    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        if weight_decay != 0 and i >= decay_idx_from:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class CustomAdamOptimizer(optim.Optimizer):

    def __init__(self, params, lr, args, H, W, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False): 

        defaults = dict(lr = lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad)
        
        super(CustomAdamOptimizer, self).__init__(params, defaults)
        self.args = args
        self.H, self.W = H, W

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            f_custom_adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   self.H, 
                   self.W,
                   self.args,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps']
            )
        return loss