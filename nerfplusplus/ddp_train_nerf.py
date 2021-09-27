import torch
import torch.nn as nn
import torch.optim
import torch.distributed
import torch.multiprocessing
import os
import time
from data_loader_split import load_data_split
from collections import OrderedDict
import numpy as np
import wandb
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, TINY_NUMBER
import logging
import json

from config_argparser import config_parser
from create_nerf import create_nerf

import sys
sys.path.append("..")

from model.reprojection import (
    runSuperGlueSinglePair, init_superglue, image_pair_candidates
)
from model.ray_dist_loss import *
from nerf_sample_ray_split import render_ray_from_camera

logger = logging.getLogger(__package__)

def setup_logger():
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception("""
        Not all your cameras are bounded by the unit sphere; please make sure 
        the cameras are normalized properly!
        """)
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER      # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00       # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
    denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def render_single_image(rank, world_size, models, ray_sampler, chunk_size, camera_model, camera_idx=None):
    ##### parallel rendering of a single image
    
    with torch.no_grad():
        if camera_idx is not None:
            ray_batch = ray_sampler.get_all(camera_model, camera_idx, None, rank)
        else:
            ray_batch = ray_sampler.get_all(camera_model, None, ray_sampler, rank)
    
    if (ray_batch['ray_d'].shape[0] // world_size) * world_size != ray_batch['ray_d'].shape[0]:
        raise Exception('Number of pixels in the image is not divisible by the number of GPUs!\n\t# pixels: {}\n\t# GPUs: {}'.format(ray_batch['ray_d'].shape[0],
                                                                                                                                     world_size))
    
    # split into ranks; make sure different processes don't overlap
    rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)[rank].to(rank)

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]

        dots_sh = list(ray_d.shape[:-1])
        for m in range(models['cascade_level']):
            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples,]).expand(dots_sh + [N_samples,]).to(rank)

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])    # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]                              # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)    # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=True)    # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                del bg_weights
                del bg_depth_mid
                del bg_depth_samples
                torch.cuda.empty_cache()

            with torch.no_grad():
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0)

    # merge results from different processes
    if rank == 0:
        ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                # generate tensors to store results from other processes
                sh = list(ret_merge_chunk[m][key].shape[1:])
                ret_merge_rank[m][key] = [torch.zeros(*[size,]+sh, dtype=torch.float32) for size in rank_split_sizes]
                torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
                ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0).reshape(
                                            (ray_sampler.H, ray_sampler.W, -1)).squeeze()
                # print(m, key, ret_merge_rank[m][key].shape)
    else:  # send results to main process
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                torch.distributed.gather(ret_merge_chunk[m][key])

    # only rank 0 program returns
    if rank == 0:
        return ret_merge_rank
    else:
        return None


def log_view_to_tb(global_step, log_data, gt_img, mask, prefix=''):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    
    image_dict = {}
    
    image_dict[prefix + "rgb_gt"] = wandb.Image(rgb_im)

    for m in range(len(log_data)):
        rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        image_dict[prefix + 'level_{}/rgb'.format(m)] = wandb.Image(rgb_im)

        rgb_im = img_HWC2CHW(log_data[m]['fg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        image_dict[prefix + 'level_{}/fg_rgb'.format(m)] = wandb.Image(rgb_im)

        depth = log_data[m]['fg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        image_dict[prefix + 'level_{}/fg_depth'.format(m)] = wandb.Image(depth_im)

        rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        image_dict[prefix + 'level_{}/bg_rgb'.format(m)] = wandb.Image(rgb_im)

        depth = log_data[m]['bg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        image_dict[prefix + 'level_{}/bg_depth'.format(m)] = wandb.Image(depth_im)

        bg_lambda = log_data[m]['bg_lambda']
        bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='hot', append_cbar=True,
                                            mask=mask))
        image_dict[prefix + 'level_{}/bg_lambda'.format(m)] = wandb.Image(bg_lambda_im)
    
    wandb.log(image_dict, step=global_step)

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    # port = np.random.randint(12355, 12399)
    # os.environ['MASTER_PORT'] = '{}'.format(port)
    os.environ['MASTER_PORT'] = str(args.master_addr)
    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()

def get_images(ray_samplers):
    return [ray.get_img() for ray in ray_samplers]

def ddp_train_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size, args)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### Create log dir and copy the config file
    if rank == 0:
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        f = os.path.join(args.basedir, args.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(args.basedir, args.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())
    torch.distributed.barrier()

    ray_samplers, camera_info = load_data_split(args.datadir, args.scene, split='train',
                                   try_load_min_depth=args.load_min_depth, args=args)

    if args.run_fisheye:
        render = "train"
    else:
        render = "validation"
    val_ray_samplers, _ = load_data_split(args.datadir, args.scene, split=render,
                                       try_load_min_depth=args.load_min_depth, skip=args.testskip, 
                                       args=args)

    if args.use_camera:
        matcher = init_superglue(args, rank)
        extrinsics = camera_info["extrinsics"].numpy()
        image_pair_cache = {}
        with torch.no_grad():
            feasible_image_pairs = image_pair_candidates(extrinsics, args)
        images = get_images(ray_samplers)

    # write training image names for autoexposure
    if args.optim_autoexpo:
        f = os.path.join(args.basedir, args.expname, 'train_images.json')
        with open(f, 'w') as file:
            img_names = [ray_samplers[i].img_path for i in range(len(ray_samplers))]
            json.dump(img_names, file, indent=2)

    ###### create network and wrap in ddp; each process should do this
    start, models, camera_model = create_nerf(rank, args, camera_info)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777)

    ##### only main process should do the logging

    if rank == 0:
        wandb.init(
            config=args,
            name=args.expname,
            project="NeRF++"
        )

    # start training
    optim = models["optim"]

    what_val_to_log = 0             # helper variable for parallel rendering of a image
    what_train_to_log = 0

    for global_step in range(start+1, start+1+args.N_iters):

        # Curriculum Learning

        decay_rate = args.lrate_decay_factor
        decay_steps = args.lrate_decay_steps * 1000
        new_lrate = max(
            args.lrate * (decay_rate ** (global_step / decay_steps)), args.lrate * 1e-2
        )
        for param_group in optim.param_groups:
            param_group["lr"] = new_lrate

        activate_ie = global_step == args.add_ie and args.use_camera and \
            hasattr(camera_model, "intrinsics_noise") and \
                hasattr(camera_model, "extrinsics_noise")
        activate_radial = global_step == args.add_radial and args.use_camera and \
            hasattr(camera_model, "distortion_noise")
        activate_od = global_step == args.add_od and args.use_camera and \
            hasattr(camera_model, "ray_o_noise") and \
                hasattr(camera_model, "ray_d_noise")

        if activate_ie:
            camera_model.intrinsics_noise.requires_grad_(True)
            camera_model.extrinsics_noise.requires_grad_(True)
            logger.info("Activate learnable intrinsic and extrinsic")

        if activate_radial:
            camera_model.distortion_noise.requires_grad_(True)
            logger.info("Activate learnable radial distortion")

        if activate_od:
            camera_model.ray_o_noise.requires_grad_(True)
            camera_model.ray_d_noise.requires_grad_(True)
            logger.info("Activate learnable rayo rayd distortion")

        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        scalars_to_log['resolution'] = ray_samplers[0].resolution_level
        scalars_to_log["lr"] = new_lrate
        # randomly sample rays and move to device
        img_i = np.random.randint(low=0, high=len(ray_samplers))
        all_rets = []                                  # results on different cascade levels
        loss = 0.0
        optim.zero_grad()
        for m in range(models['cascade_level']):

            if m == 0: 
                ray_batch, select_inds = ray_samplers[img_i].random_sample(
                    args.N_rand, camera_model, img_i, rank
                )
            else: 
                ray_batch, _ = ray_samplers[img_i].random_sample(
                    args.N_rand, camera_model, img_i, rank, select_inds=select_inds
                )

            # forward and backward
            dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
            
            net = models['net_{}'.format(m)]

            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])  # [...,]
                fg_near_depth = ray_batch['min_depth']  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                fg_depth = perturb_samples(fg_depth)   # random perturbation during training

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                            [1, ] * len(dots_sh) + [N_samples,]).expand(dots_sh + [N_samples,]).to(rank)
                bg_depth = perturb_samples(bg_depth)   # random perturbation during training
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])    # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]                              # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=False)    # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=False)    # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

            
            ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth, img_name=ray_batch['img_name'])
            all_rets.append(ret)

            rgb_gt = ray_batch['rgb'].to(rank)
            if 'autoexpo' in ret:
                scale, shift = ret['autoexpo']
                scalars_to_log['level_{}/autoexpo_scale'.format(m)] = scale.item()
                scalars_to_log['level_{}/autoexpo_shift'.format(m)] = shift.item()
                # rgb_gt = scale * rgb_gt + shift
                rgb_pred = (ret['rgb'] - shift) / scale
                rgb_loss = img2mse(rgb_pred, rgb_gt)
                loss = loss + rgb_loss + args.lambda_autoexpo * (torch.abs(scale-1.)+torch.abs(shift))
            else:
                rgb_loss = img2mse(ret['rgb'], rgb_gt)
                loss = loss + rgb_loss

            if m == models['cascade_level'] - 1 and \
                global_step % args.alternate_frequency == 0 \
                    and args.use_camera \
                        and global_step > args.add_prd and args.add_prd != -1 \
                            and not args.run_fisheye:
                img_j = np.random.choice(feasible_image_pairs[img_i])
                pair_key = (img_i, img_j)

                if pair_key in image_pair_cache.keys():
                    result = image_pair_cache[pair_key]
                else:
                    with torch.no_grad():
                        result = runSuperGlueSinglePair(
                            matcher,
                            images[img_i],
                            images[img_j],
                            rank,
                            args
                        )
                    result = preprocess_match(result)
                    image_pair_cache[pair_key] = result
                    
                kps0_list, kps1_list = result

                kps0_list_flatten = (kps0_list[:, 1] * camera_info["W"] + kps0_list[:, 0]).long()
                kps1_list_flatten = (kps1_list[:, 1] * camera_info["W"] + kps1_list[:, 0]).long()

                kps0_list = kps0_list + 0.5
                kps1_list = kps1_list + 0.5

                rays_i = render_ray_from_camera(
                    camera_model, img_i, kps0_list_flatten, rank
                )

                rays_j = render_ray_from_camera(
                    camera_model, img_j, kps1_list_flatten, rank
                )

                prd_loss, num_matches = proj_ray_dist_loss_single(
                    kps0_list=kps0_list,
                    kps1_list=kps1_list,
                    img_idx0=img_i,
                    img_idx1=img_j,
                    rays0=(rays_i[0], rays_i[1]),
                    rays1=(rays_j[0], rays_j[1]),
                    mode="train",
                    device=rank, 
                    H=camera_info["H"],
                    W=camera_info["W"],
                    args=args,
                    camera_model=camera_model,
                    method="NeRF++",
                    i_map=np.array(range(len(ray_samplers)))
                )

                scalars_to_log["initial_matches"] = len(kps0_list)
                if not torch.isnan(prd_loss):
                    loss = loss + args.ray_dist_loss_weight * prd_loss
                    scalars_to_log["num_matches"] = num_matches
                    scalars_to_log["prd_loss"] = prd_loss
            
            scalars_to_log['level_{}/loss'.format(m)] = rgb_loss.item()
            scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(rgb_loss.item())
        
        loss.backward()
        optim.step()

        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        ### only main process should do the logging
        if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
            logstr = '{} step: {} '.format(args.expname, global_step)
            if not camera_model is None and (
                global_step % args.camera_log == 0 or global_step < 10
            ):
                noise_to_log, image_to_log = camera_model.log_noises(
                    camera_info["intrinsics"].to(rank), 
                    camera_info["extrinsics"].to(rank)
                )
                scalars_to_log.update(noise_to_log)
                for k, v in image_to_log.items():
                    wandb.log({k: wandb.Image(v)}, step=global_step)

            wandb.log(scalars_to_log, step=global_step)
            logger.info(logstr)

        ### each process should do this; but only main process merges the results
        if global_step % args.i_img == 0 or global_step == start+1:
            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)
            log_data = render_single_image(rank, args.world_size, models, val_ray_samplers[idx], args.chunk_size, camera_model)
            what_val_to_log += 1
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Logged a random validation view in {} seconds'.format(dt))
                log_view_to_tb(global_step, log_data, gt_img=val_ray_samplers[idx].get_img(), mask=None, prefix='val/')

            time0 = time.time()
            idx = what_train_to_log % len(ray_samplers)
            log_data = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size, camera_model)
            what_train_to_log += 1
            dt = time.time() - time0
            if rank == 0:   # only main process should do this
                logger.info('Logged a random training view in {} seconds'.format(dt))
                log_view_to_tb(global_step, log_data, gt_img=ray_samplers[idx].get_img(), mask=None, prefix='train/')

            del log_data
            torch.cuda.empty_cache()    

        if rank == 0 and (global_step % args.i_weights == 0):
            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()
            
            name = 'optim'
            to_save[name] = models[name].state_dict()

            for m in range(models['cascade_level']):
                name = 'net_{}'.format(m)
                to_save[name] = models[name].state_dict()

            if camera_model is not None:
                to_save["camera_model"] = camera_model.state_dict()
            torch.save(to_save, fpath)

    # clean up for multi-processing
    cleanup()


def train():
    parser = config_parser()
    args = parser.parse_args() 
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_train_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    train()


