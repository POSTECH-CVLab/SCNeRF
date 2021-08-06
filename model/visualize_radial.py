import argparse
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt 

from matplotlib import cm

cmap = cm.get_cmap('hsv')


def visualize(args):
    pth = torch.load(args.ft_path)
    k1, k2, k3 = pth["camera_model"]["distortion"]

    H, W, _ = pth["camera_model"]["ray_o_noise"].shape

    k1, k2, k3 = k1.item(), k2.item(), k3.item()
    radial = np.array([[(j + 0.5, i + 0.5) for j in range(W * 20)] for i in range(H * 20)])
    radial_dist = (radial - np.array([[W * 10, H * 10]])) / np.array([[W * 10, H * 10]])

    r2 = radial_dist ** 2
    r2 = r2[:, :, 0] + r2[:, :, 1]

    vis = r2 * k1 + r2 ** 2 * k2 + r2 **3 * k3
    norm = vis / np.abs(vis).max()

    ori = np.array([[-(np.arctan(radial_dist[i,j,1] / radial_dist[i,j,0]) ) for j in range(W*20)] for i in range(H*20)])
    ori[:, 0:W*10] = np.pi + ori[:, 0:W*10]
    ori[H*10:H*20, W*10:W*20] = 2 * np.pi + ori[H*10:H*20, W*10:W*20]
    ori = ori / ori.max()

    color = np.array([[cmap(ori[i][j]) for j in range(W*20)]for i in range(H * 20)])

    norm = norm ** 0.7
    color_dist = color * norm[:, :, None]

    img = Image.fromarray((color_dist[:, :, :3] * 255).astype(np.uint8))
    img.save(args.output_fig)

    img = Image.fromarray((color[:, :, :3] * 255).astype(np.uint8))
    img.save("colormap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_path", type=str, required=True)
    parser.add_argument("--output_fig", type=str, required=True)
    args = parser.parse_args()
    visualize(args)
