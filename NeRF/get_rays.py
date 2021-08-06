import torch
import numpy as np


def get_rays_full_image_no_camera(H, W, focal, extrinsic):

    # We don't need to use multiple extrinsic.
    assert extrinsic.dim() == 2

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()

    dirs = torch.stack(
        [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)],
        dim=-1)
    kps_list = torch.stack([i, j, torch.ones_like(i)], -1).long()

    rays_d = torch.sum(dirs[..., np.newaxis, :] * extrinsic[:3, :3], -1)
    rays_o = extrinsic[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_full_image_use_camera(
    H,
    W,
    camera_model,
    idx_in_camera_param=None,
    extrinsic=None,
):
    # idx_in_camera_param: index of corresponding extrinsic parameters
    # in camera_model. It can be either single int or list of ints.
    # Extrinsics is none when generating rays in train mode.
    # Extrinsics is not none when it is in evaluation mode.

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()

    kps_list = torch.stack([i, j, torch.ones_like(i)],
                           -1).long().reshape(-1, 3)

    intrinsics_inv = torch.inverse(camera_model.get_intrinsic()[:3, :3])
    extrinsic = camera_model.get_extrinsic()[idx_in_camera_param, :3, :3] \
        if extrinsic is None else extrinsic

    dirs = torch.einsum("ij, kj -> ik", kps_list.float(), intrinsics_inv)
    dirs[:, 1:3] = -dirs[:, 1:3]

    if extrinsic.dim() == 3:
        rays_d = torch.sum(dirs[..., np.newaxis, :] * extrinsic[:, :3, :3], -1)
        rays_o = extrinsic[:, :3, -1]
    else:
        rays_d = torch.sum(dirs[..., np.newaxis, :] * extrinsic[:3, :3], -1)
        rays_o = extrinsic[:3, -1].expand(rays_d.shape)

    if hasattr(camera_model, "ray_o_noise"):
        ray_o_noise = camera_model.get_ray_o_noise().reshape(H, W, 3)
        ray_o_noise_kps = ray_o_noise[kps_list[:, 1], kps_list[:, 0]]
        rays_o = rays_o + ray_o_noise_kps

    if hasattr(camera_model, "ray_d_noise"):
        ray_d_noise = camera_model.get_ray_d_noise().reshape(H, W, 3)
        ray_d_noise_kps = ray_d_noise[kps_list[:, 1], kps_list[:, 0]]

        rays_d = rays_d + ray_d_noise_kps
        rays_d = rays_d / (rays_d.norm(dim=1)[:, None] + 1e-10)

    return rays_o, rays_d


def get_rays_kps_no_camera(H, W, focal, extrinsic, kps_list):

    # kps_list must follow (x, y, 1) form.
    assert kps_list[:, 0].max() < W
    assert kps_list[:, 1].max() < H
    assert extrinsic.dim() == 2

    kps_list = kps_list.long()
    # Rotate ray directions from camera frame to the world frame
    dirs = torch.stack(
        [(kps_list[:, 0] - W * .5) / focal, -(kps_list[:, 1] - H * .5) / focal,
         -torch.ones_like(kps_list[:, 0])], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * extrinsic[:3, :3], -1)
    rays_o = extrinsic[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_kps_use_camera(H,
                            W,
                            camera_model,
                            kps_list,
                            idx_in_camera_param=None,
                            extrinsic=None):

    # idx_in_camera_param: index of corresponding extrinsic parameters
    # in camera_model. It can be either single int or list of ints.
    # Selecting keypoints are only used in train mode.
    # kps_list has the form of (x, y)

    assert kps_list[:, 0].max() < W
    assert kps_list[:, 1].max() < H
    assert (
        idx_in_camera_param is None and not extrinsic is None or \
        not idx_in_camera_param is None and extrinsic is None
        )

    kps_list_expand = torch.stack(
        [kps_list[:, 0], kps_list[:, 1],
         torch.ones_like(kps_list[:, 0])],
        dim=-1).float()



    intrinsics_inv = torch.inverse(camera_model.get_intrinsic()[:3, :3])
    extrinsic = camera_model.get_extrinsic()[idx_in_camera_param] \
        if extrinsic is None else extrinsic

    dirs = torch.einsum("ij, kj -> ik", kps_list_expand.float(),
                        intrinsics_inv)
    dirs[:, 1:3] = -dirs[:, 1:3]

    if extrinsic.dim() == 3:
        rays_d = torch.sum(dirs[..., np.newaxis, :] * extrinsic[:, :3, :3], -1)
        rays_o = extrinsic[:, :3, -1]
    else:
        rays_d = torch.sum(dirs[..., np.newaxis, :] * extrinsic[:3, :3], -1)
        rays_o = extrinsic[:3, -1].expand(rays_d.shape)

    if hasattr(camera_model, "ray_o_noise"):
        kps_list = kps_list.long()
        ray_o_noise = camera_model.get_ray_o_noise().reshape(H, W, 3)
        ray_o_noise_kps = ray_o_noise[kps_list[:, 1], kps_list[:, 0]]
        rays_o = rays_o + ray_o_noise_kps

    if hasattr(camera_model, "ray_d_noise"):
        kps_list = kps_list.long()
        ray_d_noise = camera_model.get_ray_d_noise().reshape(H, W, 3)
        ray_d_noise_kps = ray_d_noise[kps_list[:, 1], kps_list[:, 0]]

        rays_d = rays_d + ray_d_noise_kps
        rays_d = rays_d / (rays_d.norm(dim=1)[:, None] + 1e-10)

    return rays_o, rays_d


def get_rays_np(H, W, focal, extrinsic):

    # Used only when camera model is None.

    W_arange = np.arange(W, dtype=np.float32)
    H_arange = np.arange(H, dtype=np.float32)

    i, j = np.meshgrid(W_arange, H_arange, indexing='xy')
    dirs = np.stack(
        [(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)

    rays_d = np.sum(dirs[..., np.newaxis, :] * extrinsic[:3, :3], -1)
    rays_o = np.broadcast_to(extrinsic[:3, -1], np.shape(rays_d))

    return rays_o, rays_d
