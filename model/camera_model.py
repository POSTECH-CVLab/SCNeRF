import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import wandb

import sys
sys.path.append("..")
from model.camera_utils import *


class CameraModel(nn.Module):
    def __init__(self, intrinsics, extrinsics, args, H, W):
        nn.Module.__init__(self)
        self.args = args
        self.H, self.W = H, W
        self.model_name = args.camera_model

        self.ray_o_noise_scale = args.ray_o_noise_scale
        self.ray_d_noise_scale = args.ray_d_noise_scale
        self.extrinsics_noise_scale = args.extrinsics_noise_scale
        self.intrinsics_noise_scale = args.intrinsics_noise_scale

    def get_ray_d_noise(self):
        return (
            nn.functional.interpolate(
                self.ray_d_noise.permute(2, 0, 1)[None, :, :],
                (self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(-1, 3)
        ) * self.ray_d_noise_scale

    def get_ray_o_noise(self):
        return (
            nn.functional.interpolate(
                self.ray_o_noise.permute(2, 0, 1)[None, :, :],
                (self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(-1, 3)
        ) * self.ray_o_noise_scale

    def get_extrinsic(self):
        raise Exception("function get_intrinsic not implemented!")

    def get_intrinsic(self):
        raise Exception("function get_extrinsic not implemented!")

    def log_noises(self, gt_intrinsic, gt_extrinsic):
        noise_to_log = {}
        image_to_log = {}
            
        noise_to_log["camera/intrinsic_noise_mean"] = \
            self.get_intrinsic().abs().mean()
        noise_to_log["camera/intrinsic_noise_std"] = \
            self.get_intrinsic().abs().mean()
        
        noise_to_log["camera/fx"] = self.get_intrinsic()[0][0]
        noise_to_log["camera/fy"] = self.get_intrinsic()[1][1]
        noise_to_log["camera/cx"] = self.get_intrinsic()[0][2]
        noise_to_log["camera/cy"] = self.get_intrinsic()[1][2]
        
        noise_to_log["camera/fx_err"] = (
            self.get_intrinsic()[0][0] - gt_intrinsic[0][0]
        ).abs()
        noise_to_log["camera/fy_err"] = (
            self.get_intrinsic()[1][1] - gt_intrinsic[1][1]
        ).abs()
        noise_to_log["camera/cx_err"] = (
            self.get_intrinsic()[0][2] - gt_intrinsic[0][2]
        ).abs()
        noise_to_log["camera/cy_err"] = (
            self.get_intrinsic()[1][2] - gt_intrinsic[1][2]
        ).abs()
            
        if hasattr(self, "extrinsics_noise"):
            
            noise_to_log["camera/extrinsic_noise_mean"] = \
                self.get_extrinsic().abs().mean()
            noise_to_log["camera/extrinsic_noise_std"] = \
                self.get_extrinsic().abs().std()
            
            noise_to_log["camera/extrinisic_err"] = (
                self.get_extrinsic() - gt_extrinsic
            ).abs().mean()
            
        if hasattr(self, "ray_o_noise"): 
            
            noise_to_log["camera/ray_o_noise_mean"] = \
                self.get_ray_o_noise().abs().mean()
            noise_to_log["camera/ray_o_noise_std"] = \
                self.get_ray_o_noise().abs().std()
            
            rgb_image = self.get_ray_o_noise().reshape(self.H, self.W, 3)
            image_to_log["camera/ray_o_noise"] = to_pil_normalize(rgb_image)

        if hasattr(self, "ray_d_noise"):
            noise_to_log["camera/ray_d_noise_mean"] = \
                self.get_ray_d_noise().abs().mean()
            noise_to_log["camera/ray_d_noise_std"] = \
                self.get_ray_d_noise().abs().std()

            rgb_image = self.get_ray_d_noise().reshape(self.H, self.W, 3)
            image_to_log["camera/ray_d_noise"] = to_pil_normalize(rgb_image)

        if hasattr(self, "distortion_noise"):

            k1, k2 = self.get_distortion()
            noise_to_log["camera/k1"] = k1
            noise_to_log["camera/k2"] = k2

        return noise_to_log, image_to_log
        

class PinholeModelRotNoiseLearning10kRayoRayd(CameraModel):
    def __init__(self, intrinsics, extrinsics, args, H, W):
        super(PinholeModelRotNoiseLearning10kRayoRayd, self).__init__(
            intrinsics, extrinsics, args, H, W
        )
        fx, fy, tx, ty = (
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
        )
        extrinsics = torch.from_numpy(np.stack(extrinsics))
        params = rotation2orth(extrinsics[:, :3, :3])
        translations = extrinsics[:, :3, 3]

        ray_o_noise = torch.zeros((H // args.grid_size, W // args.grid_size, 3))
        ray_d_noise = torch.zeros((H // args.grid_size, W // args.grid_size, 3))

        self.register_parameter(
            name="intrinsics_initial",
            param=nn.Parameter(
                torch.Tensor([fx, fy, tx, ty]), requires_grad=False
            ),
        )
        self.register_parameter(
            name="extrinsics_initial",
            param=nn.Parameter(
                torch.cat([params, translations], dim=-1), requires_grad=False
            ),
        )
        self.register_parameter(
            name="intrinsics_noise",
            param=nn.Parameter(torch.zeros(4)),
        )
        self.register_parameter(
            name="extrinsics_noise",
            param=nn.Parameter(torch.zeros_like(self.extrinsics_initial)),
        )
        self.register_parameter(
            name="ray_o_noise", param=nn.Parameter(ray_o_noise)
        )
        self.register_parameter(
            name="ray_d_noise", param=nn.Parameter(ray_d_noise)
        )
        self.multiplicative_noise = args.multiplicative_noise

    def get_intrinsic(self):
        return intrinsic_param_to_K(
            self.intrinsics_initial
            + (
                self.intrinsics_noise * 
                self.intrinsics_noise_scale * 
                self.intrinsics_initial
            )
            if self.multiplicative_noise else
            self.intrinsics_initial
            + (self.intrinsics_noise * self.intrinsics_noise_scale)
        )

    def get_extrinsic(self):
        extrinsic = get_44_rotation_matrix_from_33_rotation_matrix(
            ortho2rotation(
                self.extrinsics_initial[:, :6]
                + self.extrinsics_noise_scale * self.extrinsics_noise[:, :6]
            )
        )
        extrinsic[..., :3, 3] = (
            self.extrinsics_initial[:, 6:]
            + self.extrinsics_noise_scale * self.extrinsics_noise[:, 6:]
        )
        return extrinsic

    def forward(self, idx):
        extrinsic = get_44_rotation_matrix_from_33_rotation_matrix(
            ortho2rotation(
                self.extrinsics_initial[idx, None, :6] +
                (
                    self.extrinsics_noise_scale * 
                    self.extrinsics_noise[idx, None, :6]
                )
            )
        )
        extrinsic[..., :3, 3] = (
            self.extrinsics_initial[idx, 6:]
            + self.extrinsics_noise_scale * self.extrinsics_noise[idx, 6:]
        )
        return self.get_intrinsic(), extrinsic.squeeze()


class PinholeModelRotNoiseLearning10kRayoRaydDistortion(CameraModel):
    def __init__(self, intrinsics, extrinsics, args, H, W, k=None):
        super(PinholeModelRotNoiseLearning10kRayoRaydDistortion, self).__init__(
            intrinsics, extrinsics, args, H, W
        )
        fx, fy, tx, ty = (
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
        )
        extrinsics = torch.from_numpy(np.stack(extrinsics))
        params = rotation2orth(extrinsics[:, :3, :3])
        translations = extrinsics[:, :3, 3]

        ray_noise = torch.zeros((H // args.grid_size, W // args.grid_size, 3))

        self.register_parameter(
            name="intrinsics_initial",
            param=nn.Parameter(
                torch.Tensor([fx, fy, tx, ty]), requires_grad=False
            ),
        )
        self.register_parameter(
            name="extrinsics_initial",
            param=nn.Parameter(
                torch.cat([params, translations], dim=-1), requires_grad=False
            ),
        )
        if not k is None:
            self.register_parameter(
                name="distortion_initial", 
                param=nn.Parameter(torch.tensor([k[0], k[1]]), requires_grad=False),
            )
        else:
            self.register_parameter(
                name="distortion_initial", 
                param=nn.Parameter(torch.zeros(2), requires_grad=False),
            )

        self.register_parameter(
            name="intrinsics_noise",
            param=nn.Parameter(torch.zeros(4)),
        )
        self.register_parameter(
            name="extrinsics_noise",
            param=nn.Parameter(torch.zeros_like(self.extrinsics_initial)),
        )
        self.register_parameter(
            name="ray_o_noise", param=nn.Parameter(ray_noise)
        )
        self.register_parameter(
            name="ray_d_noise", param=nn.Parameter(ray_noise)
        )
        self.register_parameter(
            name="distortion_noise", param=nn.Parameter(torch.zeros(2))
        )
        self.multiplicative_noise = args.multiplicative_noise \
            if hasattr(args,"multiplicative_noise") else False

    def get_intrinsic(self):
        return intrinsic_param_to_K(
            self.intrinsics_initial + (
                self.intrinsics_noise_scale * 
                self.intrinsics_noise * 
                self.intrinsics_initial
            )
            if self.multiplicative_noise else
            self.intrinsics_initial
            + self.intrinsics_noise_scale * self.intrinsics_noise
        )

    def get_extrinsic(self):
        extrinsic = get_44_rotation_matrix_from_33_rotation_matrix(
            ortho2rotation(
                self.extrinsics_initial[:, :6]
                + self.extrinsics_noise_scale * self.extrinsics_noise[:, :6]
            )
        )
        extrinsic[..., :3, 3] = (
            self.extrinsics_initial[:, 6:]
            + self.extrinsics_noise_scale * self.extrinsics_noise[:, 6:]
        )
        return extrinsic

    def forward(self, idx):
        extrinsic = get_44_rotation_matrix_from_33_rotation_matrix(
            ortho2rotation(
                self.extrinsics_initial[idx, None, :6] +
                (
                    self.extrinsics_noise_scale * 
                    self.extrinsics_noise[idx, None, :6]
                )
            )
        )
        extrinsic[..., :3, 3] = (
            self.extrinsics_initial[idx, 6:]
            + self.extrinsics_noise_scale * self.extrinsics_noise[idx, 6:]
        )
        return self.get_intrinsic(), extrinsic.squeeze()

    def get_distortion(self):
        return self.distortion_initial + self.distortion_noise * \
            self.args.distortion_noise_scale
