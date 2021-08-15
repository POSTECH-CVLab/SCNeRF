import torch
import json
import os
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo
from model.camera_model import *
from custom_optim import CustomAdamOptimizer

logger = logging.getLogger(__package__)

def create_nerf(rank, args, camera_info):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is 
    # initialized with same weights at different processes
    torch.manual_seed(777)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    camera_model = None
    if args.use_camera: 
        intrinsics = camera_info["intrinsics"]
        extrinsics = camera_info["extrinsics"]
        H, W = camera_info["H"], camera_info["W"]
        if args.camera_model == "pinhole_rot_noise_10k_rayo_rayd":
            camera_model = PinholeModelRotNoiseLearning10kRayoRayd(
                intrinsics, extrinsics, args, H, W
            ).to(rank)
        else:
            camera_model = PinholeModelRotNoiseLearning10kRayoRaydDistortion(
                intrinsics, extrinsics, args, H, W, camera_info["k"]
            ).to(rank)
        
    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [
            int(x.strip()) for x in args.cascade_samples.split(',')
        ]        
    
    parameters = []
    
    for m in range(models['cascade_level']):
        img_names = None
        if args.optim_autoexpo:
            # load training image names for autoexposure
            f = os.path.join(args.basedir, args.expname, 'train_images.json')
            with open(f) as file:
                img_names = json.load(file)
        net = NerfNetWithAutoExpo(
            args, optim_autoexpo=args.optim_autoexpo, img_names=img_names
            ).to(rank)
        net = DDP(
            net, device_ids=[rank], output_device=rank, 
            find_unused_parameters=True
        )

        parameters = [*parameters, *net.parameters()]

        # net = DDP(net, device_ids=[rank], output_device=rank)
        models['net_{}'.format(m)] = net

    if camera_model is not None:
        parameters = [*parameters, *camera_model.parameters()]

    if args.use_custom_optim:
        optim = CustomAdamOptimizer(
            params=nn.ParameterList(parameters), lr=args.lrate, 
            betas=(0.9, 0.999), weight_decay=args.non_linear_weight_decay, 
            H=H, W=W, args=args
        )
    else:
        optim = torch.optim.Adam(nn.ParameterList(parameters), lr=args.lrate)
        
    models["optim"] = optim
    
    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [
            os.path.join(args.basedir, args.expname, f)
                for f in sorted(
                    os.listdir(os.path.join(args.basedir, args.expname))
                ) if f.endswith('.pth')
        ]

    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])

    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m)]:
                models[name].load_state_dict(to_load[name])
                
        pretrained_dict = to_load["optim"]
        model_dict = models["optim"].state_dict()
        model_dict["state"].update(pretrained_dict["state"])
        models["optim"].load_state_dict(model_dict)

        if args.load_camera:
            assert not args.load_test
            camera_dict = {}
            for key, val in to_load["camera_model"].items():
                if key in ["extrinsics_noise", "extrinsics_initial"]:
                    continue
                camera_dict[key] = val
            camera_origin = camera_model.state_dict()
            camera_origin.update(camera_dict)
            camera_model.load_state_dict(camera_origin)

        if args.load_test:
            assert not args.load_camera
            camera_origin = camera_model.state_dict()
            camera_origin.update(to_load["camera_model"])
            camera_model.load_state_dict(camera_origin)

    if not args.load_test:
        deactivate_ie = start < args.add_ie and args.use_camera and \
            hasattr(camera_model, "intrinsics_noise") and \
                hasattr(camera_model, "extrinsics_noise")
        deactivate_radial = start < args.add_radial and args.use_camera and \
            hasattr(camera_model, "distortion_noise")
        deactivate_od = start < args.add_od and args.use_camera and \
            hasattr(camera_model, "ray_o_noise") and \
                hasattr(camera_model, "ray_d_noise")

        if deactivate_ie:
            camera_model.intrinsics_noise.requires_grad_(False)
            camera_model.extrinsics_noise.requires_grad_(False)
            logger.info("Deactivated learnable intrinsic and extrinsic")
        
        if deactivate_radial :
            camera_model.distortion_noise.requires_grad_(False)
            logger.info("Deactivated learnable radial distortion")
        
        if deactivate_od:
            camera_model.ray_o_noise.requires_grad_(False)
            camera_model.ray_d_noise.requires_grad_(False)
            logger.info("Deactivated learnable ray offset and direction noise")

    return start, models, camera_model