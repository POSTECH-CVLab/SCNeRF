import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio

########################################################################################################################
# ray batch sampling
########################################################################################################################
def get_rays_single_image(H, W, intrinsics, c2w, k=None):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)
    
    if k is not None:
        r2 = (pixels[:2] - np.array([[W/2], [H/2]])) / (np.array([[W / 2], [H / 2]]))
        pixels[:2] = (pixels[:2] - np.array([[W/2], [H/2]])) * \
            (1 + r2**2 * k[0] + r2**4 * k[1]) + np.array([[W/2], [H/2]])
    
    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w, k,
                       img_path=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.img_path = img_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth
        self.k = k

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
                self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            self.rays_o, self.rays_d, self.depth = get_rays_single_image(
                self.H, self.W, self.intrinsics, self.c2w_mat, self.k
            )

    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_all(self, camera_model, camera_idx, ray_sampler, rank):
        
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        if camera_model is None:
            rays_o = self.rays_o
            rays_d = self.rays_d
            depth = self.depth
        elif camera_idx != None and ray_sampler is None:
            all_idx = np.arange(0, self.W_orig * self.H_orig)
            rays_o, rays_d, depth = render_ray_from_camera(
                camera_model, camera_idx, all_idx, rank
            )
        else:
            all_idx = np.arange(0, self.W_orig * self.H_orig)
            rays_o, rays_d, depth = render_ray_from_camera(
                camera_model, None, all_idx, rank, ray_sampler.c2w_mat
            )

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(
            self, N_rand, camera_model, camera_idx, rank, select_inds=None
        ):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if select_inds is None:
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
        if camera_model is None: 
            rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
            rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
            depth = self.depth[select_inds]         # [N_rand, ]
        else:
            rays_o, rays_d, depth = render_ray_from_camera(
                camera_model, camera_idx, select_inds, rank
            )

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        elif camera_model is not None:
            min_depth = 1e-4 * torch.ones_like(rays_d[..., 0])
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('rgb', rgb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path)
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k]).to(rank)

        return ret, select_inds

def render_ray(intrinsics, extrinsic, select_inds, H, W, rank):

    rays_o, rays_d, depth = get_rays_single_image(H, W, intrinsics, extrinsic)

    return rays_o[select_inds], rays_d[select_inds], depth[select_inds]

def render_ray_from_camera(
        camera_model, camera_idx, select_inds, rank, extrinsic=None
    ):    
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    W, H = camera_model.W, camera_model.H
    if not camera_idx is None:
        intrinsics, c2w = camera_model(camera_idx)
    else:
        assert extrinsic is not None
        intrinsics = camera_model.get_intrinsic()
        c2w = torch.from_numpy(extrinsic).to(rank)

    N_rand = len(select_inds)

    if isinstance(select_inds, torch.Tensor):
        select_inds = select_inds.cpu().detach().numpy()

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1)[select_inds].astype(dtype=np.float32) + 0.5
    v = v.reshape(-1)[select_inds].astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)
    pixels = torch.from_numpy(pixels).to(rank)

    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    if hasattr(camera_model, "distortion_noise"):
        (k0, k1) = camera_model.get_distortion()
        center = torch.stack([cx, cy]).view(2, -1)
        r2 = (pixels[:2] - center) / center
        pixels[:2] =  (pixels[:2] - center) * \
            (1 + r2**2 * k0 + r2**4 * k1) + center

    intrinsic_inv = torch.zeros_like(intrinsics[:3, :3])
    intrinsic_inv[[0, 1, 0, 1, 2], [0, 1, 2, 2, 2]] += torch.stack(
        [
            1.0 / intrinsics[0][0], 1.0 / intrinsics[1][1], 
            - intrinsics[0][2] / intrinsics[0][0],
            - intrinsics[1][2] / intrinsics[1][1], torch.tensor(1.0).to(rank)
        ]
    )
    
    rays_d = intrinsic_inv @ pixels
    rays_d = c2w[:3, :3] @ rays_d
    rays_d = torch.transpose(rays_d, 1, 0)

    rays_o = c2w[:3, 3].view(1, 3).repeat(N_rand, 1).to(rank)

    if hasattr(camera_model, "ray_o_noise"):
        rays_o = rays_o + camera_model.get_ray_o_noise()[select_inds].to(rank)

    if hasattr(camera_model, "ray_d_noise"):
        rays_d = rays_d + camera_model.get_ray_d_noise()[select_inds].to(rank)
        rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)

    depth = c2w.T[2, 3] * torch.ones((rays_o.shape[0],)).to(rank)

    return rays_o, rays_d, depth