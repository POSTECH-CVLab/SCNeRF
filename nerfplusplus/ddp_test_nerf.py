import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging

from piqa.ssim import SSIM
from piqa.lpips import LPIPS

import sys
sys.path.insert(0, "..")
from model.reprojection import init_superglue, runSuperGlueSinglePair,image_pair_candidates, runSIFTSinglePair
from model.ray_dist_loss import preprocess_match, proj_ray_dist_loss_single

from nerf_sample_ray_split import render_ray, render_ray_from_camera

logger = logging.getLogger(__package__)


# NeRF++ version
def projected_ray_distance_evaluation(
    images, camera_info, camera_model, args
):
    if camera_model is None:
        intrinsics = camera_info["intrinsics"]
        extrinsics = camera_info["extrinsics"]
    else:
        intrinsics = camera_model.get_intrinsic()
        extrinsics = camera_model.get_extrinsic()

    intrinsics = intrinsics.cpu().detach().numpy()
    extrinsics = extrinsics.cpu().detach().numpy()
    with torch.no_grad():
        feasible_image_pairs = image_pair_candidates(extrinsics, args)

    matcher = init_superglue(args, 0)
    H, W = camera_info["H"], camera_info["W"]

    prd_loss_list = []

    for img_i in feasible_image_pairs.keys():
        for img_j in feasible_image_pairs[img_i]:
            if img_i >= img_j:
                continue
    
            result = runSuperGlueSinglePair(matcher, images[img_i], images[img_j], 0, args)
            
            kps0_list, kps1_list = preprocess_match(result)

            kps0_list_flatten = (kps0_list[:, 1] * camera_info["W"] + kps0_list[:, 0]).long()
            kps1_list_flatten = (kps1_list[:, 1] * camera_info["W"] + kps1_list[:, 0]).long()

            if args.use_camera:

                rays_i = render_ray_from_camera(camera_model, img_i, kps0_list_flatten, 0)
                rays_j = render_ray_from_camera(camera_model, img_j, kps1_list_flatten, 0)

                prd_loss, _ = proj_ray_dist_loss_single(
                    kps0_list=kps0_list,
                    kps1_list=kps1_list,
                    img_idx0=img_i,
                    img_idx1=img_j,
                    rays0=(rays_i[0], rays_i[1]),
                    rays1=(rays_j[0], rays_j[1]),
                    mode="train",
                    device=0, 
                    H=camera_info["H"],
                    W=camera_info["W"],
                    args=args,
                    camera_model=camera_model,
                    method="NeRF++",
                    i_map=np.array(range(len(extrinsics)))
                )
                if not torch.isnan(prd_loss):
                    prd_loss_list.append(prd_loss)

            else: 
                kps0_list_flatten = kps0_list_flatten.cpu().detach().numpy()
                kps1_list_flatten = kps1_list_flatten.cpu().detach().numpy()

                rays_i = render_ray(intrinsics, extrinsics[img_i], kps0_list_flatten, H, W, 0)
                rays_j = render_ray(intrinsics, extrinsics[img_j], kps1_list_flatten, H, W, 0)
                rays_i = (torch.from_numpy(rays_i[0]).to(0), torch.from_numpy(rays_i[1]).to(0))
                rays_j = (torch.from_numpy(rays_j[0]).to(0), torch.from_numpy(rays_j[1]).to(0))
                
                prd_loss, _ = proj_ray_dist_loss_single(
                    kps0_list=kps0_list,
                    kps1_list=kps1_list,
                    img_idx0=img_i,
                    img_idx1=img_j,
                    rays0=rays_i, 
                    rays1=rays_j,
                    mode="train",
                    method="NeRF++",
                    device=0,
                    H=camera_info["H"],
                    W=camera_info["W"],
                    args=args,
                    intrinsic=torch.from_numpy(intrinsics),
                    extrinsic=torch.from_numpy(extrinsics),
                )
                if not torch.isnan(prd_loss):
                    prd_loss_list.append(prd_loss)
        
    return torch.stack(prd_loss_list).mean().item()


def ddp_test_nerf(rank, args):

    if rank == 0:
        SSIM_model = SSIM()
        LPIPS_model = LPIPS(network="vgg")
    ###### set up multi-processing
    setup(rank, args.world_size, args)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### create network and wrap in ddp; each process should do this
    
    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}'.format(split))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers, camera_info = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth, args=args)
        start, models, camera_model = create_nerf(rank, args, camera_info)
        
        train_psnr_list = []
        train_ssim_list = []
        train_lpips_list = []

        if not args.prd_only:

            for idx in range(len(ray_samplers)):
                ### each process should do this; but only main process merges the results
                fname = '{:06d}.png'.format(idx)
                if ray_samplers[idx].img_path is not None:
                    fname = os.path.basename(ray_samplers[idx].img_path)

                if os.path.isfile(os.path.join(out_dir, fname)):
                    logger.info('Skipping {}'.format(fname))
                    continue

                time0 = time.time()
                ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size, camera_model, camera_idx=idx)
                dt = time.time() - time0
                if rank == 0:    # only main process should do this
                    logger.info('Rendered {} in {} seconds'.format(fname, dt))

                    # only save last level
                    im = ret[-1]['rgb'].cpu().numpy()
                    # compute psnr if ground-truth is available
                    if ray_samplers[idx].img_path is not None:
                        gt_im = ray_samplers[idx].get_img()
                        psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                        
                        ssim = SSIM_model(
                            torch.clip(
                                torch.from_numpy(im).permute(2, 0, 1)[None, ...], 
                                0, 1,
                            ),
                            torch.from_numpy(gt_im).permute(2, 0, 1)[None, ...]
                        ).item()

                        lpips = LPIPS_model(
                            torch.clip(
                                torch.from_numpy(im).permute(2, 0, 1)[None, ...],
                                0, 1,
                            ),
                            torch.from_numpy(gt_im).permute(2, 0, 1)[None, ...],
                        ).item()

                        logger.info('{}: psnr={}'.format(fname, psnr))
                        logger.info("{}: ssim={}".format(fname, ssim))
                        logger.info("{}: lpips={}".format(fname,lpips))

                        train_psnr_list.append(psnr)
                        train_ssim_list.append(ssim)
                        train_lpips_list.append(lpips)

                    im = to8b(im)
                    imageio.imwrite(os.path.join(out_dir, fname), im)

                    im = ret[-1]['fg_rgb'].numpy()
                    im = to8b(im)
                    imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)

                    im = ret[-1]['bg_rgb'].numpy()
                    im = to8b(im)
                    imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

                    im = ret[-1]['fg_depth'].numpy()
                    im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    im = to8b(im)
                    imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

                    im = ret[-1]['bg_depth'].numpy()
                    im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    im = to8b(im)
                    imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

                torch.cuda.empty_cache()

        if rank == 0:
            images = [sample.get_img() for sample in ray_samplers]
            train_prd = projected_ray_distance_evaluation(
                images, camera_info, camera_model, args
            )

    if rank == 0:
        psnr_mean = np.mean(np.array(train_psnr_list))
        ssim_mean = np.mean(np.array(train_ssim_list))
        lpips_mean =  np.mean(np.array(train_lpips_list))
        prd_mean = train_prd
        print("PSNR: ", psnr_mean)
        print("SSIM: ", ssim_mean)
        print("LPIPS: ", lpips_mean)
        print("PRD: ", prd_mean)

        with open(args.expname + ".txt", "w") as fp:
            fp.write(f"PSNR : {str(psnr_mean)}\n")
            fp.write(f"SSIM : {str(ssim_mean)}\n")
            fp.write(f"LPIPS : {str(lpips_mean)}\n")
            fp.write(f"PRD: {str(prd_mean)}\n")

    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()

