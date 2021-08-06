import numpy as np
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

## NeRFmm Import
import thirdparty.ATE as ATE


def make_rand_axis(batch_size):
    vec = (np.random.rand(batch_size, 3) - 0.5)
    mag = np.linalg.norm(vec, 2, 1, keepdims=True)
    vec = vec/mag
    return vec

def R_axis_angle(axis, angle):
    r"""
    axis: (batch, 3)
    angle: (batch, 1), in radian

    From https://github.com/Wallacoloo/printipi/blob/master/util/
    rotation_matrix.py#L122
    Copyright (C) Edward d'Auvergne
    """
    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis[:,0:1], axis[:,1:2], axis[:,2:3]

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix.
    rot_mat = np.zeros((*axis.shape, 3)) # (batch, 3, 3)

    rot_mat[:, 0, 0:1] = x*xC + ca
    rot_mat[:, 0, 1:2] = xyC - zs
    rot_mat[:, 0, 2:3] = zxC + ys
    rot_mat[:, 1, 0:1] = xyC + zs
    rot_mat[:, 1, 1:2] = y*yC + ca
    rot_mat[:, 1, 2:3] = yzC - xs
    rot_mat[:, 2, 0:1] = zxC - ys
    rot_mat[:, 2, 1:2] = yzC + xs
    rot_mat[:, 2, 2:3] = z*zC + ca

    return rot_mat

def to_pil(array):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach()
        if len(array.shape) > 3 and array.shape[2] != 3:
            array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(array * 255))


def to_pil_normalize(array):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach()
        if len(array.shape) > 3 and array.shape[2] != 3:
            array = array.permute(1, 2, 0).numpy()
    image_array = np.uint8(
        (array - array.min()) / (array.max() - array.min()) * 255
    )
    return Image.fromarray(image_array)

def ortho2rotation(poses):
    r"""
    poses: batch x 6

    From https://github.com/chrischoy/DeepGlobalRegistration/blob/master/core
    /registration.py#L16
    Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) 
    and Wei Dong (weidong@andrew.cmu.edu)
    """

    def normalize_vector(v):
        r"""
        Batch x 3
        """
        v_mag = torch.sqrt((v ** 2).sum(1, keepdim=True))
        v_mag = torch.clamp(v_mag, min=1e-8)
        v = v / (v_mag + 1e-10)
        return v

    def cross_product(u, v):
        r"""
        u: batch x 3
        v: batch x 3
        """
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        i = i[:, None]
        j = j[:, None]
        k = k[:, None]
        return torch.cat((i, j, k), 1)

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = (u * a).sum(1, keepdim=True)
        norm2 = (u ** 2).sum(1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - proj_u2a(x, y_raw))
    z = cross_product(x, y)

    x = x[:, :, None]
    y = y[:, :, None]
    z = z[:, :, None]

    return torch.cat((x, y, z), 2)


def rotation2orth(rot):
    return torch.cat([rot[:, :, 0], rot[:, :, 1]], dim=-1)


def rot_from_angle(euler):

    batch = euler.shape[0]
    tensor_0 = torch.zeros((batch,)).to(euler.device)
    tensor_1 = torch.ones((batch,)).to(euler.device)

    AX, AY, AZ = euler[:, 0], euler[:, 1], euler[:, 2]

    RX = torch.stack(
        [
            torch.stack([tensor_1, tensor_0, tensor_0], dim=-1),
            torch.stack([tensor_0, torch.cos(AX), -torch.sin(AX)], dim=-1),
            torch.stack([tensor_0, torch.sin(AX), torch.cos(AX)], dim=-1),
        ],
        dim=-1,
    )

    RY = torch.stack(
        [
            torch.stack([torch.cos(AY), tensor_0, torch.sin(AY)], dim=-1),
            torch.stack([tensor_0, tensor_1, tensor_0], dim=-1),
            torch.stack([-torch.sin(AY), tensor_0, torch.cos(AY)], dim=-1),
        ],
        dim=-1,
    )

    RZ = torch.stack(
        [
            torch.stack([torch.cos(AZ), -torch.sin(AZ), tensor_0], dim=-1),
            torch.stack([torch.sin(AZ), torch.cos(AZ), tensor_0], dim=-1),
            torch.stack([tensor_0, tensor_0, tensor_1], dim=-1),
        ],
        dim=-1,
    )

    return torch.bmm(torch.bmm(RZ, RY), RX)

def angle_from_rot(R):
    x = -torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = -torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 2, 1] ** 2 + R[:, 2, 2] ** 2))
    z = -torch.atan2(R[:, 1, 0], R[:, 0, 0])
    return torch.stack([x, y, z], dim=1)


def get_44_rotation_matrix_from_33_rotation_matrix(m: torch.Tensor):
    out = torch.zeros((m.shape[0], 4, 4), device=m.device)
    out[:, :3, :3] = m
    out[:, 3, 3] = 1
    return out


def intrinsic_param_to_K(intrinsics):
    device = intrinsics.device
    intrinsic_mat = torch.eye(4, 4).to(device)
    intrinsic_mat[[0, 1, 0, 1], [0, 1, 2, 2]] = intrinsics
    return intrinsic_mat