import torch
import numpy as np

from model.lookup import lookup

def preprocess_match(match_result):

    match_result = match_result[0]
    kps0 = match_result["kps0"]
    kps1 = match_result["kps1"]
    matches = match_result["matches"]

    if len(matches) == 0:
        return None, None

    kps0 = torch.stack([kps0[match_[0]] for match_ in matches])
    kps1 = torch.stack([kps1[match_[1]] for match_ in matches])

    return torch.stack([kps0, kps1])


def proj_ray_dist_loss_single(
        kps0_list,
        kps1_list,
        img_idx0,
        img_idx1,
        rays0,
        rays1,
        mode,
        device,
        H,
        W,
        args,
        camera_model=None,
        intrinsic=None, 
        extrinsic=None,
        eps=1e-10,
        i_map=None, 
        method="NeRF"
    ):
    
    # extrinsic must be 

    assert mode in ["train", "val", "test"]
    assert method in ["NeRF", "NeRF++"]
    assert kps0_list[:, 0].max() < W and kps1_list[:, 0].max() < W
    assert kps0_list[:, 1].max() < H and kps1_list[:, 1].max() < H

    if mode == "train":

        if not camera_model is None:
            # In train mode, when using camera model, we use currently 
            # calibrating parameters to estimate the ray distance loss.
            # i_map is required since parameters in the camera have different
            # indexes with the indexes in extrinsic_gt parameters. 
            assert intrinsic is None
            assert extrinsic is None
            assert not i_map is None
            intrinsic = camera_model.get_intrinsic().to(device)
            extrinsic = camera_model.get_extrinsic()
            img_idx0_in_camera = np.where(i_map == img_idx0)[0][0]
            img_idx1_in_camera = np.where(i_map == img_idx1)[0][0]
            extrinsic = extrinsic[
                [img_idx0_in_camera, img_idx1_in_camera]
            ].to(device)
        
        else:
            # In train mode, without camera model, we use the parameters
            # with noise to estimate the ray distance loss. 
            assert not intrinsic is None
            assert not extrinsic is None
            assert isinstance(intrinsic, torch.Tensor)
            assert isinstance(extrinsic, torch.Tensor)
            intrinsic = intrinsic.to(device)
            extrinsic = extrinsic[[img_idx0, img_idx1]].to(device)
            
    else:
        
        if not camera_model is None:
            # In val/test mode with camera model, we use the GT extrinsic 
            # parameters and calibrated parameters (intrinsic, rayo, rayd noise)
            assert intrinsic is None
            assert not extrinsic is None
            intrinsic = camera_model.get_intrinsic().to(device)
            extrinsic = extrinsic[[[img_idx0, img_idx1]]].to(device)
        
        else:
            # In val/test mode without camera model, we use all the GT
            # parameters to estimate ray dist loss.
            assert not intrinsic is None
            assert not extrinsic is None 
            intrinsic = intrinsic.to(device)
            extrinsic = extrinsic[[[img_idx0, img_idx1]]].to(device)

    rays0_o, rays0_d = rays0
    rays1_o, rays1_d = rays1
    rays0_o, rays0_d = rays0_o.unsqueeze(0), rays0_d.unsqueeze(0)
    rays1_o, rays1_d = rays1_o.unsqueeze(0), rays1_d.unsqueeze(0)

    intrinsic = intrinsic.clone()
    if method == "NeRF":
        # NeRF is using a different coordinate system.
        intrinsic[0][0] = -intrinsic[0][0]
    
    extrinsic_inv = torch.zeros_like(extrinsic)
    extrinsic_rot_inv = extrinsic[:, :3, :3].transpose(1, 2)
    extrinsic_inv[:, :3, :3] = extrinsic_rot_inv
    extrinsic_inv[:, :3, 3] = - (
        extrinsic_rot_inv @ extrinsic[:, :3, 3, None]
        ).squeeze(-1)
    extrinsic_inv[:, 3, 3] = 1.

    rays0_d = rays0_d / (rays0_d.norm(p=2, dim=-1)[:, :, None] + eps)
    rays1_d = rays1_d / (rays1_d.norm(p=2, dim=-1)[:, :, None] + eps) 

    rays0_o_world = torch.cat([
        rays0_o, torch.ones((rays0_o.shape[:2]), device=device)[:, :, None]
    ], dim=-1)[:, :, :3]

    rays1_o_world = torch.cat([
        rays1_o, torch.ones((rays1_o.shape[:2]), device=device)[:, :, None]
    ], dim=-1)[:, :, :3]

    rays0_d_world = rays0_d[:, :, :3]
    rays1_d_world = rays1_d[:, :, :3]

    r0_r1 = torch.einsum(
        "ijk, ijk -> ij", 
        rays0_d_world, 
        rays1_d_world
    )
    t0 = (
        torch.einsum(
            "ijk, ijk -> ij", 
            rays0_d_world, 
            rays0_o_world - rays1_o_world
        ) - r0_r1
        * torch.einsum(
            "ijk, ijk -> ij", 
            rays1_d_world, 
            rays0_o_world - rays1_o_world
        )
    ) / (r0_r1 ** 2 - 1 + eps)

    t1 = (
        torch.einsum(
            "ijk, ijk -> ij", 
            rays1_d_world, 
            rays1_o_world - rays0_o_world
        ) - r0_r1
        * torch.einsum(
            "ijk, ijk -> ij", 
            rays0_d_world, 
            rays1_o_world - rays0_o_world
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

    p0_proj_to_im1 = torch.einsum("ijk, pk -> ijp", p0_4d, extrinsic_inv[1])
    p1_proj_to_im0 = torch.einsum("ijk, pk -> ijp", p1_4d, extrinsic_inv[0])
    p0_norm_im1 = torch.einsum("ijk, pk -> ijp", p0_proj_to_im1, intrinsic)
    p1_norm_im0 = torch.einsum("ijk, pk -> ijp", p1_proj_to_im0, intrinsic)
    
    p0_norm_im1_2d = p0_norm_im1[:, :, :2] / \
        (p0_norm_im1[:, :, 2, None] + eps)
    p1_norm_im0_2d = p1_norm_im0[:, :, :2] / \
        (p1_norm_im0[:, :, 2, None] + eps)

    # Chirality check: remove rays behind cameras
    # First, flatten the correspondences
    # Find indices of valid rays
    valid_t0 = (t0 > 0).flatten()
    valid_t1 = (t1 > 0).flatten()
    valid = torch.logical_and(valid_t0, valid_t1)

    p1_norm_im0_2d, kps0_list = p1_norm_im0_2d[0, valid], kps0_list[valid]
    p0_norm_im1_2d, kps1_list = p0_norm_im1_2d[0, valid], kps1_list[valid]

    # if camera_model is not None and hasattr(camera_model, "distortion"):
    
    #     valid_p1, p0_norm_im1_2d = lookup(
    #         W, H, camera_model.distortion, p0_norm_im1_2d[:, 0], p0_norm_im1_2d[:, 1], device
    #     )
    #     valid_p0, p1_norm_im0_2d = lookup(
    #         W, H, camera_model.distortion, p1_norm_im0_2d[:, 0], p1_norm_im0_2d[: ,1], device
    #     )
    #     valid = torch.logical_and(valid_p0, valid_p1)
    
    #     p1_norm_im0_2d, kps0_list = p1_norm_im0_2d[valid], kps0_list[valid]
    #     p0_norm_im1_2d, kps1_list = p0_norm_im1_2d[valid], kps1_list[valid]

    # Second, select losses that are valid

    loss0_list = (
        (p1_norm_im0_2d - kps0_list) ** 2
    ).sum(-1).flatten()
    loss1_list = (
        (p0_norm_im1_2d - kps1_list) ** 2
    ).sum(-1).flatten()
    
    if mode == "train":

        loss0_valid_idx = torch.logical_and(
            loss0_list < args.proj_ray_dist_threshold, 
            torch.isfinite(loss0_list)
        )
        loss1_valid_idx = torch.logical_and(
            loss1_list < args.proj_ray_dist_threshold, 
            torch.isfinite(loss1_list)
        )
        loss0 = loss0_list[loss0_valid_idx].mean()
        loss1 = loss1_list[loss1_valid_idx].mean()

        num_matches = torch.logical_and(
            loss0_valid_idx, loss1_valid_idx
        ).float().sum().item() 
        
        return 0.5 * (loss0 + loss1), num_matches
        
    else:
        loss0_invalid_idx = torch.logical_or(
            loss0_list > args.proj_ray_dist_threshold,
            torch.logical_not(torch.isfinite(loss0_list))
        )
        loss0_list[loss0_invalid_idx] = args.proj_ray_dist_threshold
        loss0 = loss0_list.mean()

        loss1_invalid_idx = torch.logical_or(
            loss1_list > args.proj_ray_dist_threshold,
            torch.logical_not(torch.isfinite(loss1_list))
        )
        loss1_list[loss1_invalid_idx] = args.proj_ray_dist_threshold
        loss1 = loss1_list.mean()
        
        del intrinsic

        return 0.5 * (loss0 + loss1), None