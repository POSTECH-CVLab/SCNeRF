from PIL import Image
import numpy as np
import torch


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
    return Image.fromarray(np.uint8((array - array.min()) / (array.max() - array.min()) * 255))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")