from reprojection import runSuperGlueSinglePair,image_pair_candidates, runSIFTSinglePair
from ray_dist_loss import preprocess_match, proj_ray_dist_loss_single
import torch
import numpy as np

import os

from random import random
import numpy as np
import torch
import torchvision.transforms as TF

import matplotlib.pyplot as plt

tol=1e-4
match_num = 4
run_unit_test = lambda args, kwargs, test_name: None if not args.debug else \
    test_name(**kwargs)


def unit_test_matches(**kwargs):
    
    msg = "Failed to pass the unit test named matches"
    print("Starting Unit Test : matches")
    
    dirname = "_unit_test_matches_result"
    
    # Check whether argument is currently provided. 
    assert "args" in kwargs.keys(), msg
    assert "result" in kwargs.keys(), msg 
    assert "img_i" in kwargs.keys(), msg
    assert "img_j" in kwargs.keys(), msg
    assert "img_i_idx" in kwargs.keys(), msg
    assert "img_j_idx" in kwargs.keys(), msg
    
    args= kwargs["args"]
    result = kwargs["result"]
    img_i, img_j = kwargs["img_i"], kwargs["img_j"]
    img_i_idx, img_j_idx = kwargs["img_i_idx"], kwargs["img_j_idx"]
    kps1, kps2 = result
    W = img_i.shape[1]
    
    # Draw matches and save them
    assert hasattr(args, "datadir"), msg    
    scene_name = args.datadir.split("/")[-1]
    scene_path = os.path.join(dirname, scene_name)
    os.makedirs(scene_path, exist_ok=True)
    img_name = "{}_{}.png".format(img_i_idx, img_j_idx)
    img_path = os.path.join(scene_path, img_name)
    
    img_cat = torch.cat([img_i, img_j], dim=1)
    img_cat_pil = TF.ToPILImage()(img_cat.permute(2, 0, 1))
    plt.imshow(img_cat_pil)
    
    i_visualize = np.random.choice(range(len(kps1)), match_num)
    
    for i in i_visualize: 
        kp1, kp2 = kps1[i].cpu().numpy(), kps2[i].cpu().numpy()
        color = (random(), random(), random())
        plt.plot([kp1[0], kp2[0]+W], [kp1[1], kp2[1]], c=color, lw=2)
        
    plt.savefig(img_path)    
    plt.close()


def projected_ray_distance_evaluation(
        images, 
        index_list,
        args,
        ray_fun,
        ray_fun_gt,
        H,
        W,
        mode,
        matcher,
        gt_intrinsic,
        gt_extrinsic,
        method, 
        device,
        intrinsic=None, 
        extrinsic=None,
        camera_model=None, 
        i_map=None, 
    ):
    
    prd_list = []

    match_fun = runSuperGlueSinglePair if args.matcher == "superglue" else \
        runSIFTSinglePair

    extrinsic_gt_numpy = gt_extrinsic[index_list].cpu().numpy()
    
    with torch.no_grad():
        feasible_image_pairs = image_pair_candidates(
            extrinsic_gt_numpy, args, index_list
        )

    for img_i in feasible_image_pairs.keys(): 
        for img_j in feasible_image_pairs[img_i]:
            
            if img_i >= img_j:
                continue
            
            result = match_fun(
                matcher,
                images[img_i],
                images[img_j],
                0,
                args
            )
            
            kps0_list, kps1_list = preprocess_match(result)
            if kps0_list is None and kps1_list is None:
                continue
            result = kps0_list, kps1_list
            kwargs_unit_test = {
                "args": args,
                "result": result,
                "img_i": images[img_i],
                "img_j": images[img_j],
                "img_i_idx": img_i,
                "img_j_idx": img_j
            }
            run_unit_test(
                args, kwargs_unit_test, unit_test_matches
            )
            if mode != "train": 
            # Acquiring correct matches using the ground truth camera info
            # In the training mode, we don't use the ground truth information.
                
                rays_i_gt = ray_fun_gt(
                    H=H, W=W,focal=gt_intrinsic[0][0], 
                    extrinsic=gt_extrinsic[img_i], kps_list=kps0_list
                )
                rays_j_gt = ray_fun_gt(
                    H=H, W=W,focal=gt_intrinsic[0][0], 
                    extrinsic=gt_extrinsic[img_j], kps_list=kps1_list
                )

                filter_idx = filter_matches_with_gt(
                    kps0_list=kps0_list,
                    kps1_list=kps1_list,
                    H=H,
                    W=W,
                    gt_intrinsic=gt_intrinsic,
                    gt_extrinsic=gt_extrinsic[[img_i, img_j]],
                    rays0=rays_i_gt,
                    rays1=rays_j_gt,
                    args=args,
                    device=device,
                    method=method
                )
                kps0_list = kps0_list[filter_idx]
                kps1_list = kps1_list[filter_idx]

            if camera_model is None:
                
                # Evaluate with gt_extrinsic for val,test
                # Evaluate with noisy_extrinsic for train
                extrinsic_evaluate = gt_extrinsic if mode != "train" else \
                    extrinsic
                
                rays_i = ray_fun(
                    H=H, W=W, focal=intrinsic[0][0], 
                    extrinsic=extrinsic_evaluate[img_i], kps_list=kps0_list
                )
                rays_j = ray_fun(
                    H=H, W=W, focal=intrinsic[0][0], 
                    extrinsic=extrinsic_evaluate[img_j], kps_list=kps1_list
                )
                
                projected_ray_dist, _ = proj_ray_dist_loss_single(
                    kps0_list=kps0_list, kps1_list=kps1_list, img_idx0=img_i,
                    img_idx1=img_j, rays0=rays_i, rays1=rays_j, mode=mode, 
                    device=device, H=H, W=W, args=args, 
                    intrinsic=gt_intrinsic, extrinsic=extrinsic_evaluate
                )
            
            else:
                
                # In the train mode, we use the
                extrinsic_evaluate = gt_extrinsic if mode != "train" else \
                    None
                extrinsic_evaluate_i = gt_extrinsic[img_i] if mode != "train" \
                    else None
                extrinsic_evaluate_j = gt_extrinsic[img_j] if mode != "train" \
                    else None
                camera_idx_i = np.where(i_map == img_i)[0][0] \
                    if mode == "train" else None
                camera_idx_j = np.where(i_map == img_j)[0][0] \
                    if mode == "train" else None
                
                rays_i = ray_fun(
                    H=H, W=W, camera_model=camera_model,
                    extrinsic=extrinsic_evaluate_i, kps_list=kps0_list, 
                    idx_in_camera_param=camera_idx_i
                )
                rays_j = ray_fun(
                    H=H, W=W, camera_model=camera_model,
                    extrinsic=extrinsic_evaluate_j, kps_list=kps1_list, 
                    idx_in_camera_param=camera_idx_j
                )
                
                projected_ray_dist, _ = proj_ray_dist_loss_single(
                    kps0_list=kps0_list, kps1_list=kps1_list, img_idx0=img_i,
                    img_idx1=img_j, rays0=rays_i, rays1=rays_j, mode=mode, 
                    device=device, H=H, W=W, args=args, i_map=i_map,
                    camera_model=camera_model, extrinsic=extrinsic_evaluate
                )

            if not torch.isnan(projected_ray_dist): 
                prd_list.append(projected_ray_dist.item())
    
    prd_list = torch.tensor(prd_list)

    return prd_list.mean()


# Since SuperGlue sometimes fail to acquire reliable matches, 
# we filter matches using the ground truth information only when
# evaluating PRD on val/test.

def filter_matches_with_gt(
    kps0_list,
    kps1_list,
    W,
    H,
    gt_intrinsic,
    gt_extrinsic,
    rays0,
    rays1,
    args,
    method,
    device,
    eps=1e-6
):
    assert method in ["NeRF", "NeRF++"]
    assert kps0_list.dim() == 2 and kps1_list.dim() == 2

    gt_intrinsic=gt_intrinsic.clone().detach()
    # NeRF is using an opposite coordinate. 
    if method == "NeRF":
        gt_intrinsic[0][0] = -gt_intrinsic[0][0]

    rays0_o, rays0_d = rays0
    rays1_o, rays1_d = rays1
    rays0_o, rays0_d = rays0_o.unsqueeze(0), rays0_d.unsqueeze(0)
    rays1_o, rays1_d = rays1_o.unsqueeze(0), rays1_d.unsqueeze(0)

    gt_extrinsic_inv = torch.inverse(gt_extrinsic.cpu())
    gt_extrinsic_inv = gt_extrinsic_inv.to(device)

    rays0_d = rays0_d / (rays0_d.norm(p=2, dim=-1)[:, :, None] + eps)
    rays1_d = rays1_d / (rays1_d.norm(p=2, dim=-1)[:, :, None] + eps)

    rays0_o_world = torch.cat(
        [
            rays0_o, 
            torch.ones((rays0_o.shape[:2]), device=device)[:, :, None]
        ], 
        dim=-1
    )[:, :, :3]
    rays1_o_world = torch.cat(
        [
            rays1_o, 
            torch.ones((rays1_o.shape[:2]), device=device)[:, :, None]
        ], 
        dim=-1
    )[:, :, :3]

    rays0_d_world = rays0_d[:, :, :3]
    rays1_d_world = rays1_d[:, :, :3]

    r0_r1 = torch.einsum("ijk, ijk -> ij", rays0_d_world, rays1_d_world)
    t0 = (
        torch.einsum(
            "ijk, ijk -> ij", rays0_d_world, rays0_o_world - rays1_o_world
        ) - r0_r1
        * torch.einsum(
            "ijk, ijk -> ij", rays1_d_world, rays0_o_world - rays1_o_world
        )
    ) / (r0_r1 ** 2 - 1 + eps)

    t1 = (
        torch.einsum(
            "ijk, ijk -> ij", rays1_d_world, rays1_o_world - rays0_o_world
        ) - r0_r1
        * torch.einsum(
            "ijk, ijk -> ij", rays0_d_world, rays1_o_world - rays0_o_world
        )
    ) / (r0_r1 ** 2 - 1 + eps)

    p0 = t0[:, :, None] * rays0_d_world + rays0_o_world
    p1 = t1[:, :, None] * rays1_d_world + rays1_o_world

    p0_4d = torch.cat(
        [p0, torch.ones((p0.shape[:2]), device=device)[:, :, None]], dim=-1
    )
    p1_4d = torch.cat(
        [p1, torch.ones((p1.shape[:2]), device=device)[:, :, None]], dim=-1
    )

    p0_proj_to_im1 = torch.einsum(
        "ijk, ipk -> ijp", p0_4d, gt_extrinsic_inv[1:]
    )
    p1_proj_to_im0 = torch.einsum(
        "ijk, ipk -> ijp", p1_4d, gt_extrinsic_inv[:-1]
    )
    p0_norm_im1 = torch.einsum("ijk, pk -> ijp", p0_proj_to_im1, gt_intrinsic)
    p1_norm_im0 = torch.einsum("ijk, pk -> ijp", p1_proj_to_im0, gt_intrinsic)

    p0_norm_im1_2d = p0_norm_im1[:, :, :2] / (p0_norm_im1[:, :, 2, None] + eps)
    p1_norm_im0_2d = p1_norm_im0[:, :, :2] / (p1_norm_im0[:, :, 2, None] + eps)

    # Chirality check: remove rays behind cameras
    # First, flatten the correspondences
    # Find indices of valid rays
    valid_t0 = (t0 > 0).flatten()
    valid_t1 = (t1 > 0).flatten()
    valid = torch.logical_and(valid_t0, valid_t1)

    # Second, select losses that are valid
    # When using NeRF++
    loss0_list = ((p1_norm_im0_2d - kps0_list) ** 2).sum(-1).flatten()
    loss1_list = ((p0_norm_im1_2d - kps1_list) ** 2).sum(-1).flatten()

    # Remove cloned tensor after the computation
    del gt_intrinsic
    
    return torch.logical_and(
        torch.logical_and(loss0_list < 1.0, loss1_list < 1.0), valid
    )