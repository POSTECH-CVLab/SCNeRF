import os
import sys
import imageio
import time
import random
import socket

import cv2 as cv

from tqdm import tqdm, trange
from datetime import datetime

from run_nerf_helpers import (
    fix_seeds,
    img2mse,
    mse2psnr
)

# Ray generating functions
from get_rays import (
    get_rays_full_image_no_camera,
    get_rays_full_image_use_camera,
    get_rays_kps_no_camera,
    get_rays_kps_use_camera,
    get_rays_np
)

from render import (
    render,
    render_path
)

# Argument Parser
from config_argparse import config_parser

# DataLoader
from load_llff import load_llff_data
from load_blender import load_blender_data

# NeRF network
from create_nerf import create_nerf

# SSIM, LPIPS from piqa library
from piqa.ssim import SSIM
from piqa.lpips import LPIPS

###################### Importing from ".." ######################
# Add a path to use our code
sys.path.insert(0, "..")
sys.path.insert(0, "../thirdparty/nerfmm/")

from src.utils import (
    str2bool,
    to_pil,
    to_pil_normalize
)

from model.camera_model import *
from model.reprojection import (
    runSIFTSinglePair, runSuperGlueSinglePair, init_superglue, image_pair_candidates
)
from model.ray_dist_loss import preprocess_match, proj_ray_dist_loss_single

from thirdparty.superglue.models.matching import Matching
from thirdparty.nerfmm.utils.align_traj import align_ate_c2b_use_a2b

######################## Unit Tests #############################
from unit_tests.noise_injection_test import unit_test_noise_injection
from unit_tests.visualize_matches import unit_test_matches

#################################################################

from prd_evaluation import projected_ray_distance_evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_unit_test = lambda args, kwargs, test_name: None if not args.debug else \
    test_name(**kwargs)

SSIM_model = SSIM().cuda()
LPIPS_model = LPIPS(network="vgg").cuda()

def train():
    parser = config_parser()
    args = parser.parse_args()

    fix_seeds(args.seed)
    if args.matcher == "superglue":
        matcher = init_superglue(args, 0)
    elif args.matcher == "sift":
        matcher = cv.SIFT_create()
    
    # Debug Mode
    if args.debug:
        args.expname = "delete-me"
        args.precrop_iters = 0
        
    host_name = socket.gethostname()
    date_time =  datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.expname = "{}_{}_{}".format(args.expname, host_name, date_time)

    if not args.debug: 
        wandb.init(
            name=args.expname,
            project="SCN",
            entity="nextgennerf"
        )

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load datacre
    if args.dataset_type == 'llff':
        
        (
            images, noisy_extrinsic, bds, render_poses, i_test, gt_camera_info
        ) = load_llff_data(
            args.datadir, args.factor, recenter=True, bd_factor=.75, 
            spherify=args.spherify, args=args
        )
        
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        
        noisy_idx = i_train[0]
        hwf = noisy_extrinsic[noisy_idx, :3, -1]
        noisy_extrinsic = noisy_extrinsic[:, :3, :4]
        noisy_extrinsic_new = np.zeros(
            (len(noisy_extrinsic), 4, 4)
        ).astype(np.float32)
        noisy_extrinsic_new[:, :3, :] = noisy_extrinsic
        noisy_extrinsic_new[:, 3, 3] = 1
        noisy_extrinsic = noisy_extrinsic_new

        (gt_intrinsic, gt_extrinsic) = gt_camera_info
        
        print("Loaded LLFF dataset")
        print("Images shape : {}".format(images.shape))
        print("HWF : {}".format(hwf))
        print("Directory path of data : {}".format(args.datadir))

        print('DEFINING BOUNDS')
        
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

    elif args.dataset_type == 'blender':
        (
            images, noisy_extrinsic, render_poses, hwf, i_split, gt_camera_info
        ) = load_blender_data(args.datadir, args.half_res, args, args.testskip)
        
        print("Loaded blender dataset")
        print("Images shape : {}".format(images.shape))
        print("HWF : {}".format(hwf))
        print("Directory path of data : {}".format(args.datadir))
        
        (gt_intrinsic, gt_extrinsic) = gt_camera_info
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + \
                (1. - images[..., -1:])
        else:
            images = images[..., :3]
            
    else:
        assert False,"Invalid Dataset Selected"
        return
    
    noisy_train_poses = noisy_extrinsic[i_train]

    if args.ray_loss_type != "none":
        image_pair_cache = {}
        with torch.no_grad():
            image_pairs = \
                image_pair_candidates(noisy_train_poses, args, i_train)

    # Cast intrinsics to right types
    H, W, noisy_focal = hwf
    H, W =int(H), int(W)
    hwf = [H, W, noisy_focal]
    
    # When running with nerfmm setup, fx = W, fy = H
    if args.run_without_colmap:
        noisy_initial_intrinsic = torch.tensor(
            [
                [W, 0, W/2, 0],
                [0, H, H/2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
    else:
        noisy_initial_intrinsic = torch.tensor(
            [
                [noisy_focal, 0, W/2, 0],
                [0, noisy_focal, H/2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    (
        render_kwargs_train, render_kwargs_test, start, 
        grad_vars, optimizer, camera_model,
    ) = create_nerf(
        args, noisy_focal, noisy_train_poses, H, W, mode="train", device=device
    )

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = render_poses.to(device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    use_batching = not args.no_batching
    
    if use_batching:
        
        # Precomputing rays_rgb is not required when camera_model is not None.
        if camera_model is None:
            rays = np.stack(
                [get_rays_np(H, W, noisy_focal, p) \
                 for p in noisy_extrinsic[:, :3, :4]],
                axis=0
            )
            rays_rgb = np.concatenate([rays, images[:, None]], 1)
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
            rays_rgb = rays_rgb.astype(np.float32)
        
        shuffled_ray_idx = np.arange(len(i_train) * H * W)
        np.random.shuffle(shuffled_ray_idx)
        shuffled_image_idx = shuffled_ray_idx // (H * W)
        
        if camera_model is None:
            rays_rgb = rays_rgb[shuffled_ray_idx]
            
        i_batch = 0

    # Move training data to GPU
    images = torch.tensor(images).to(device)
    noisy_extrinsic = torch.tensor(noisy_extrinsic).to(device)
    if use_batching and camera_model is None:
        rays_rgb = torch.tensor(rays_rgb).to(device)

    N_iters = 200000 + 1 if not args.debug else 2
    N_iters = args.N_iters if args.N_iters is not None else N_iters
    print(f"N-iters: {N_iters}")
    
    print("TRAIN views are {}".format(i_train))
    print("VAL views are {}".format(i_val))
    print("TEST views are {}".format(i_test))

    start = start + 1
    for i in trange(start, N_iters):
        
        if i == start and i < args.add_ieod and camera_model is not None:
            camera_model.intrinsics_noise.requires_grad_(False)
            camera_model.extrinsics_noise.requires_grad_(False)
            camera_model.ray_o_noise.requires_grad_(False)
            camera_model.ray_d_noise.requires_grad_(False)
            print("Deactivated learnable ieod")

        if i == args.add_ieod:
            camera_model.intrinsics_noise.requires_grad_(True)
            camera_model.extrinsics_noise.requires_grad_(True)
            camera_model.ray_o_noise.requires_grad_(True)
            camera_model.ray_d_noise.requires_grad_(True)
            print("Activated learnable ieod")


        time0 = time.time()
        scalars_to_log = {}
        images_to_log = {}
    
        # Sample random ray batch
        if use_batching and camera_model is None:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        elif use_batching and not camera_model is None:
            
            ### shuffled_image_idx = shuffled_ray_idx // (H * W)
            
            image_idx_curr_step = shuffled_image_idx[i_batch:i_batch + N_rand]
            h_list = shuffled_ray_idx[i_batch:i_batch + N_rand] % (H * W) // W
            w_list = shuffled_ray_idx[i_batch:i_batch + N_rand] % (H * W) % W
            
            select_coords = np.stack([w_list, h_list], -1)
            assert select_coords[:, 0].max() < W
            assert select_coords[:, 1].max() < H
            
            image_idx_curr_step_tensor = torch.from_numpy(
                image_idx_curr_step
            ).cuda()
            kps_list = torch.from_numpy(select_coords).cuda()
                
            rays_o, rays_d = get_rays_kps_use_camera(
                H=H, 
                W=W, 
                camera_model=camera_model, 
                idx_in_camera_param=image_idx_curr_step_tensor,
                kps_list=kps_list
            )

            batch_rays = torch.stack([rays_o, rays_d])
            index_train = i_train[
                shuffled_image_idx[i_batch: i_batch + N_rand]
            ]
            target_s = images[index_train, h_list, w_list]

            img_i = np.random.choice(index_train)
            img_i_train_idx = np.where(i_train == img_i)[0][0]

            i_batch += N_rand
            if i_batch >= len(shuffled_ray_idx):
                print("Shuffle data after an epoch!")
                np.random.shuffle(shuffled_ray_idx)
                shuffled_image_idx = shuffled_ray_idx // (H * W)
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            img_i_train_idx = np.where(i_train == img_i)[0][0]
            target = images[img_i]
            noisy_pose = noisy_extrinsic[img_i, :3, :4]

            if N_rand is not None:

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(
                                W // 2 - dW, W // 2 + dW - 1, 2 * dW
                            ),
                            torch.linspace(
                                H // 2 - dH, H // 2 + dH - 1, 2 * dH
                            ),
                        ), 
                        -1
                    )
                    
                    if i == start:
                        print(
                            "[Config] Center cropping until iter {}".format(
                                args.precrop_iters
                            )
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, W - 1, W),
                            torch.linspace(0, H - 1, H), 
                        ),
                        -1
                    )

                coords = torch.reshape(coords, [-1, 2])
                assert coords[:, 0].max() < W and coords[:, 1].max() < H
                select_inds = np.random.choice(
                    coords.shape[0], 
                    size=[N_rand], 
                    replace=False
                ) 
                select_coords = coords[select_inds].long()  

                if camera_model is None:
                    rays_o, rays_d = get_rays_kps_no_camera(
                        H=H,
                        W=W,
                        focal=noisy_focal,
                        extrinsic=noisy_pose, 
                        kps_list=select_coords
                    )

                else:
                    rays_o, rays_d = get_rays_kps_use_camera(
                        H=H,
                        W=W,
                        camera_model=camera_model,
                        idx_in_camera_param=img_i_train_idx,
                        kps_list=select_coords
                    )

                # (2, N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)  
                # (N_rand, 3)
                target_s = target[select_coords[:, 1], select_coords[:, 0]] 

        #####  Core optimization loop  #####
        if camera_model is None:
            rgb, disp, acc, extras = render(
                H=H, W=W, chunk=args.chunk, noisy_focal=noisy_focal, 
                rays=batch_rays, verbose=i < 10, retraw=True, 
                mode="train", **render_kwargs_train,
            )
        else:
            rgb, disp, acc, extras = render(
                H=H, W=W, chunk=args.chunk, rays=batch_rays, 
                verbose=i < 10, retraw=True, camera_model=camera_model, 
                mode="train", **render_kwargs_train,
            )
            
        optimizer.zero_grad()
        train_loss_1 = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        train_psnr_1 = mse2psnr(train_loss_1)
        train_loss = train_loss_1

        match_fun = runSuperGlueSinglePair if args.matcher == "superglue" else \
            runSIFTSinglePair

        if 'rgb0' in extras:
            train_loss_0 = img2mse(extras['rgb0'], target_s)
            train_loss = train_loss + train_loss_0
            train_psnr_0 = mse2psnr(train_loss_0)

        if args.ray_loss_type != "none" and (
                global_step % args.i_ray_dist_loss == 1 or 
                args.i_ray_dist_loss == 1 or
                args.debug
            ) and global_step >= args.add_prd:
            if img_i in image_pairs.keys():
                img_j = np.random.choice(image_pairs[img_i])
                img_j_train_idx = np.where(i_train == img_j)[0][0]
                pair_key = (img_i_train_idx, img_j_train_idx)
                if pair_key in image_pair_cache.keys():
                    result = image_pair_cache[pair_key]
                else:
                    with torch.no_grad():
                        result = match_fun(
                                matcher,
                                images[img_i],
                                images[img_j],
                                0,
                                args
                            )
                        result = preprocess_match(result)
                        if result[0] is not None and result[1] is not None:
                            image_pair_cache[pair_key] = result

                    if result[0] is not None and result[1] is not None:
                        kps0_list, kps1_list = result

                        rays_i = get_rays_kps_use_camera(
                            H=H, 
                            W=W, 
                            camera_model=camera_model, 
                            idx_in_camera_param=img_i_train_idx, 
                            kps_list=kps0_list
                        )
                        rays_j = get_rays_kps_use_camera(
                            H=H,
                            W=W,
                            camera_model=camera_model,
                            idx_in_camera_param=img_j_train_idx,
                            kps_list=kps1_list
                        )

                        if camera_model is None:
                            ray_dist_loss_ret, n_match = proj_ray_dist_loss_single(
                                kps0_list=kps0_list,
                                kps1_list=kps1_list,
                                img_idx0=img_i_train_idx,
                                img_idx1=img_j_train_idx,
                                rays0=rays_i,
                                rays1=rays_j,
                                mode="train",
                                device=device,
                                H=H,
                                W=W, 
                                args=args,
                                intrinsic=noisy_initial_intrinsic,
                                extrinsic=torch.from_numpy(
                                    noisy_extrinsic
                                ).to(device),
                                method="NeRF"
                            )
                
                        else:
                            ray_dist_loss_ret, n_match = proj_ray_dist_loss_single(
                                kps0_list=kps0_list,
                                kps1_list=kps1_list,
                                img_idx0=img_i,
                                img_idx1=img_j,
                                rays0=rays_i,
                                rays1=rays_j,
                                mode="train",
                                device=device,
                                H=H,
                                W=W, 
                                args=args,
                                camera_model=camera_model,
                                method="NeRF",
                                i_map=i_train
                            )
                        
                        logger_ray_dist_key = "train/ray_dist_loss"
                        logger_n_match_key = "train/n_match"
                        logger_ray_dist_weight_key = "train/ray_dist_loss_weight"

                        scalars_to_log[logger_ray_dist_weight_key] = args.ray_dist_loss_weight
                        scalars_to_log[logger_ray_dist_key] = \
                            ray_dist_loss_ret.item()
                        scalars_to_log[logger_n_match_key] = n_match

                        train_loss = train_loss \
                            + args.ray_dist_loss_weight * ray_dist_loss_ret
                    
                train_loss.backward()
                optimizer.step()

        else:
            train_loss.backward()
            optimizer.step()
            
        if not camera_model is None and global_step % 2000 == 1:
            scalar_dict, image_dict = camera_model.log_noises(
                gt_intrinsic, 
                gt_extrinsic[i_train],
            )             
            scalars_to_log.update(scalar_dict)
            images_to_log.update(image_dict)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': (
                    render_kwargs_train['network_fn'].state_dict()
                ),
                'network_fine_state_dict': (
                    render_kwargs_train['network_fine'].state_dict()
                ),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if args.camera_model != "none":
                save_dict["camera_model"] = camera_model.state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
        
        test_render = (i % args.i_testset == 0 and i > 0) or args.debug
        val_render = (i % args.i_img == 0) or args.debug
            
        if not camera_model is None and test_render:

            gt_transformed_pose_test = align_ate_c2b_use_a2b(
                gt_extrinsic[i_train], 
                camera_model.get_extrinsic().detach(), 
                gt_extrinsic[i_test]
            )

        if not camera_model is None and val_render: 

            gt_transformed_pose_val = align_ate_c2b_use_a2b(
                gt_extrinsic[i_train], 
                camera_model.get_extrinsic().detach(), 
                gt_extrinsic[i_val]
            )
            
        # Test Rendering
        if test_render:
        # if False:

            print("Starts Test Rendering")
            with torch.no_grad():
                testsavedir = os.path.join(
                    basedir, expname, 'testset_{:06d}'.format(i)
                )
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', noisy_extrinsic[i_test].shape)

                if camera_model is None:
                    eval_prd = projected_ray_distance_evaluation(
                        images=images, 
                        index_list=i_test,
                        args=args, 
                        ray_fun=get_rays_kps_no_camera,
                        ray_fun_gt=get_rays_kps_no_camera,
                        H=H,
                        W=W,
                        mode="test",
                        matcher=matcher,
                        gt_intrinsic=gt_intrinsic,
                        gt_extrinsic=gt_extrinsic,
                        method="NeRF",
                        device=device,
                        intrinsic=gt_intrinsic,
                        extrinsic=gt_extrinsic
                    )
                    
                else:
                    eval_prd = projected_ray_distance_evaluation(
                        images=images, 
                        index_list=i_test,
                        args=args, 
                        ray_fun=get_rays_kps_use_camera,
                        ray_fun_gt=get_rays_kps_no_camera,
                        H=H,
                        W=W,
                        mode="test",
                        matcher=matcher,
                        gt_intrinsic=gt_intrinsic,
                        gt_extrinsic=gt_extrinsic,
                        method="NeRF",
                        device=device,
                        camera_model=camera_model,
                        intrinsic=gt_intrinsic,
                        extrinsic=gt_extrinsic
                    )
                    
                scalars_to_log["test/proj_ray_dist_loss"] = eval_prd
                print(f"Test projection ray distance loss {eval_prd}")
                
                with torch.no_grad():
            
                    # transform_align: 4 X 4
                    # gt_test: N X 4 X 4
                        
                    if camera_model is None:
                        _hwf = (hwf[0], hwf[1], None)
                        rgbs, disps = render_path(
                            gt_extrinsic[i_test],
                            _hwf, args.chunk, render_kwargs_test, 
                            gt_imgs=images[i_test], 
                            savedir=testsavedir, mode="test", 
                            args=args, gt_intrinsic=gt_intrinsic, 
                            gt_extrinsic=gt_extrinsic,
                            i_map=i_test
                        )
                        
                    else:                    
                        # Convert the focal to None (Not needed)
                        _hwf = (hwf[0], hwf[1], None)
                        rgbs, disps = render_path(
                            gt_transformed_pose_test,
                            _hwf, args.chunk, render_kwargs_test, 
                            gt_imgs=images[i_test], 
                            savedir=testsavedir, mode="test", 
                            camera_model=camera_model, args=args, 
                            gt_intrinsic=gt_intrinsic, 
                            gt_extrinsic=gt_extrinsic,
                            i_map=i_test,
                            transform_align=gt_transformed_pose_test
                        )

                test_psnr_list, test_ssim_list, test_lpips_list = [], [], []

                for idx in range(len(i_test)):
                    viewing_rgbs, viewing_disps = rgbs[idx], disps[idx]
                    target = images[i_test[idx]]
                    
                    viewing_rgbs = viewing_rgbs.reshape(H, W, 3)
                    viewing_disps = viewing_disps.reshape(H, W)

                    test_img_loss = img2mse(
                        torch.from_numpy(viewing_rgbs), target.detach().cpu()
                    )
                    test_psnr = mse2psnr(test_img_loss)
                    test_psnr = test_psnr.item()

                    test_ssim = SSIM_model(
                        torch.clip(
                            torch.from_numpy(
                                viewing_rgbs
                            ).permute(2, 0, 1)[None, ...], 
                            0, 
                            1,
                        ).cuda(), 
                        target.permute(2, 0, 1)[None, ...]
                    ).item()

                    test_lpips = LPIPS_model(
                        torch.clip(
                            torch.from_numpy(
                                viewing_rgbs
                            ).permute(2, 0, 1)[None, ...],
                            0, 
                            1,
                        ).cuda(), 
                        target.permute(2, 0, 1)[None, ...]
                    ).item()

                    test_psnr_list.append(test_psnr)
                    test_ssim_list.append(test_ssim)
                    test_lpips_list.append(test_lpips)

                test_psnr_mean = torch.tensor(test_psnr_list).mean().item()
                test_ssim_mean = torch.tensor(test_ssim_list).mean().item()
                test_lpips_mean = torch.tensor(test_lpips_list).mean().item()

                scalars_to_log["test/psnr"] = test_psnr_mean
                scalars_to_log["test/ssim"] = test_ssim_mean
                scalars_to_log["test/lpips"] = test_lpips_mean
                    
                print('Saved test set')
                print("[Test] PSNR: {}".format(test_psnr_mean))
                print("[Test] SSIM: {}".format(test_ssim_mean))
                print("[Test] LPIPS: {}".format(test_lpips_mean))

        if (i % args.i_print == 0):
            
                print(
                    "[TRAIN] Iter: {} Loss: {}  PSNR: {}".format(
                        i, train_loss.item(), train_psnr_1.item()
                    )
                )

                scalars_to_log["train/level_1_psnr"] = train_psnr_1.item()
                scalars_to_log["train/level_1_loss"] = train_loss_1.item()
                scalars_to_log["train/loss"] = train_loss.item()

                if "rgb0" in extras:
                    scalars_to_log["train/level_0_psnr"] = train_psnr_0.item()
                    scalars_to_log["train/level_0_loss"] = train_loss_0.item()
         
        # Validation
        if val_render:
            print("Starts Validation Rendering")
            img_i = np.random.choice(i_val)
            img_i_idx = np.where(i_val == img_i)
            target = images[img_i]

            # TODO
            with torch.no_grad():
                if camera_model is None:
                    rgb, disp, acc, extras = render(
                        H=H, W=W, chunk=args.chunk, gt_intrinsic=gt_intrinsic,
                        gt_extrinsic=gt_extrinsic, mode="val", image_idx=img_i,
                        **render_kwargs_test
                    )
                else:
                    rgb, disp, acc, extras = render(
                        H=H, W=W, chunk=args.chunk, gt_intrinsic=gt_intrinsic,
                        gt_extrinsic=gt_extrinsic, mode="val", i_map=i_val,
                        image_idx=img_i, camera_model=camera_model, 
                        transform_align=gt_transformed_pose_val[img_i_idx[0][0]], 
                        **render_kwargs_test
                    )

            rgb = rgb.reshape(H, W, 3)
            disp = disp.reshape(H, W)

            val_img_loss = img2mse(rgb, target)
            val_psnr = mse2psnr(val_img_loss)
            
            images_to_log["val/rgb"] = to_pil(rgb)
            images_to_log["val/disp"] = to_pil(disp)
            scalars_to_log["val/psnr"] = val_psnr.item()
            scalars_to_log["val/loss"] = val_img_loss.item()

            print("VAL PSNR {}: {}".format(img_i, val_psnr.item()))

            if camera_model is None:
                eval_prd = projected_ray_distance_evaluation(
                    images=images, 
                    index_list=i_val,
                    args=args, 
                    ray_fun=get_rays_kps_no_camera,
                    ray_fun_gt=get_rays_kps_no_camera,
                    H=H,
                    W=W,
                    mode="val",
                    matcher=matcher,
                    gt_intrinsic=gt_intrinsic,
                    gt_extrinsic=gt_extrinsic,
                    method="NeRF",
                    device=device,
                    intrinsic=gt_intrinsic,
                    extrinsic=gt_extrinsic,
                )

            else:
                eval_prd = projected_ray_distance_evaluation(
                    images=images, 
                    index_list=i_val,
                    args=args, 
                    ray_fun=get_rays_kps_use_camera,
                    ray_fun_gt=get_rays_kps_no_camera,
                    H=H,
                    W=W,
                    mode="val",
                    matcher=matcher,
                    gt_intrinsic=gt_intrinsic,
                    gt_extrinsic=gt_extrinsic,
                    method="NeRF",
                    device=device,
                    camera_model=camera_model
                )            
            
            scalars_to_log["val/proj_ray_dist_loss"] = eval_prd

            print("Validation PRD : {}".format(eval_prd))

        # Logging Step
            
        for key, val in images_to_log.items():
            scalars_to_log[key] = wandb.Image(val)    
        wandb.log(scalars_to_log)
                
        global_step += 1

    # Train Rendering
    
    print("Training Done")
    print("Starts Train Rendering")

    train_log_at_end = {}
    
    if camera_model is None:
        train_prd = projected_ray_distance_evaluation(
            images=images, 
            index_list=i_train,
            args=args, 
            ray_fun=get_rays_kps_no_camera,
            ray_fun_gt=get_rays_kps_no_camera,
            H=H,
            W=W,
            mode="train",
            matcher=matcher,
            gt_intrinsic=gt_intrinsic,
            gt_extrinsic=gt_extrinsic,
            method="NeRF",
            device=device,
            intrinsic=noisy_initial_intrinsic,
            extrinsic=noisy_extrinsic,
        )
        train_log_at_end["train_last/PRD"] = train_prd

    else:
        train_prd = projected_ray_distance_evaluation(
            images=images, 
            index_list=i_train,
            args=args, 
            ray_fun=get_rays_kps_use_camera,
            ray_fun_gt=get_rays_kps_no_camera,
            H=H,
            W=W,
            mode="train",
            matcher=matcher,
            gt_intrinsic=gt_intrinsic,
            gt_extrinsic=gt_extrinsic,
            method="NeRF",
            device=device,
            camera_model=camera_model,
            i_map=i_train
        )
        train_log_at_end["train_last/PRD"] = train_prd
    
    train_savedir = os.path.join(
        basedir, expname, 'trainset'
    )
    os.makedirs(train_savedir, exist_ok=True)

    with torch.no_grad():
        if camera_model is None:
            # render_kwargs_test
            rgbs, disps = render_path(
                render_poses=noisy_extrinsic[i_train], 
                noisy_extrinsic=noisy_extrinsic[i_train],
                hwf=hwf, 
                chunk=args.chunk,
                render_kwargs=render_kwargs_train, 
                mode="train", 
                gt_imgs=images[i_train], 
                savedir=train_savedir, 
                args=args, 
            )
        else:
            hwf_removed_focal = (hwf[0], hwf[1], None)
            rgbs, disps = render_path(
                render_poses=camera_model.get_extrinsic(), 
                noisy_extrinsic=camera_model.get_extrinsic(), 
                hwf=hwf_removed_focal, 
                chunk=args.chunk, 
                render_kwargs=render_kwargs_train, 
                mode="train",
                gt_imgs=images[i_train], 
                savedir=train_savedir,
                camera_model=camera_model,
                args=args, 
                i_map=i_train
            )
            
    train_psnr_list, train_ssim_list, train_lpips_list = [], [], []
    for idx in range(len(rgbs)):
        viewing_rgbs, viewing_disps = rgbs[idx], disps[idx]
        target = images[i_train[idx]]

        viewing_rgbs = viewing_rgbs.reshape(H, W, 3)
        viewing_disps = viewing_disps.reshape(H, W)

        train_img_loss = img2mse(
            torch.from_numpy(viewing_rgbs), target.detach().cpu()
        )
        train_psnr = mse2psnr(train_img_loss).item()

        train_ssim = SSIM_model(
            torch.clip(
                torch.from_numpy(
                    viewing_rgbs
                ).permute(2, 0, 1)[None, ...], 
                0, 
                1, 
            ).cuda(),
            target.permute(2, 0, 1)[None, ...]
        ).item()

        train_lpips = LPIPS_model(
            torch.clip(
                torch.from_numpy(
                    viewing_rgbs
                ).permute(2, 0, 1)[None, ...],
                0, 
                1,
            ).cuda(), 
            target.permute(2, 0, 1)[None, ...]
        ).item()

        print("PSNR{}: {}".format(i_train[idx], train_psnr))
        print("SSIM{}: {}".format(i_train[idx], train_ssim))
        print("LPIPS{}: {}".format(i_train[idx], train_lpips))

        train_psnr_list.append(train_psnr)
        train_ssim_list.append(train_ssim)
        train_lpips_list.append(train_lpips)

    train_psnr_mean = torch.tensor(train_psnr_list).mean().item()
    train_ssim_mean = torch.tensor(train_ssim_list).mean().item()
    train_lpips_mean = torch.tensor(train_lpips_list).mean().item()

    train_log_at_end["train_last/PSNR"] = train_psnr_mean
    train_log_at_end["train_last/SSIM"] = train_ssim_mean
    train_log_at_end["train_last/LPIPS"] = train_lpips_mean

    print(f"Train PSNR: {train_psnr_mean}")
    print(f"Train SSIM: {train_ssim_mean}")
    print(f"Train LPIPS: {train_lpips_mean}")

    if not args.debug:
        wandb.log(train_log_at_end, step=global_step)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
