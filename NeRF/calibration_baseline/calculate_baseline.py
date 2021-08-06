

from scipy.io import savemat
import json
import argparse
import numpy as np
from PIL import Image
import os
import sys
import torch
import cv2
import tqdm
sys.path.insert(0, "../..")

import torchvision.transforms as tf

from submodule.SuperGluePretrainedNetwork.models.matching import Matching
from model.reprojection import runSuperGlueSinglePair
from scipy.optimize import least_squares

def mendonca(intrinsic_initial, fundamental, extrinsics):
    
    intrinsic_initial = intrinsic_initial.cpu().numpy()
    intrinsic_initial = [*intrinsic_initial, 0]

    def fun(intrinsic_initial):
        ret = []

        fx, fy, cx, cy, val = intrinsic_initial
        intrinsic_mat = np.array(
                [
                    [fx, val, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ]
        )

        cnt = 0

        for i in fundamental.keys():
            for j in fundamental[i].keys():
                if i >= j:
                    continue
                cnt += 1

        intrinsic_inv = np.linalg.inv(intrinsic_mat)
        for i in fundamental.keys():
            for j in fundamental[i].keys():
                if i >= j:
                    continue
                fmat = fundamental[i][j]
                essential = intrinsic_mat.T @ fmat @ intrinsic_mat
                u, s, vh = np.linalg.svd(essential)
                r1, r2 = sorted(s)[2], sorted(s)[1]
                ret.append((r1 - r2) / (r2 + r1) / cnt)
        
        return np.array(ret)

    ret = least_squares(fun, np.array(intrinsic_initial), xtol=1e-10, method="lm")
    return ret.x

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def classical_kruppa(intrinsic_initial, fundamental, extrinsics):

    intrinsic_initial = intrinsic_initial.cpu().numpy()
    intrinsic_initial = [*intrinsic_initial, 0]

    def fun(intrinsic_initial):
        ret = []
        fx, fy, cx, cy, val = intrinsic_initial
        intrinsic_mat = np.array(
                [
                    [fx, val, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ]
        )
        W_inv = intrinsic_mat @ intrinsic_mat.T
        intrinsic_inv = np.linalg.inv(intrinsic_mat)
        for i in fundamental.keys():
            for j in fundamental[i].keys():
                if i >= j:
                    continue
                A = fundamental[i][j] @ W_inv @ fundamental[i][j].T
                A = A / np.linalg.norm(A, ord="fro")
                u, s, vh = np.linalg.svd(fundamental[i][j].T)
                epipole = vh[-1]
                epi = skew(epipole)

                B = epi @ W_inv @ epi.T 
                B = B / np.linalg.norm(B, ord="fro")

                E = A - B
                ret.append(np.concatenate([E[0, 0:3].reshape(-1), E[1, 1:3].reshape(-1)]))

        return np.array(ret).reshape(-1)

    ret = least_squares(fun, np.array(intrinsic_initial), method="lm", xtol=1e-10, ftol=1e-10)
    return ret.x

def simple_kruppa(intrinsic_initial, fundamental, extrinsics):

    intrinsic_initial = intrinsic_initial.cpu().numpy()
    intrinsic_initial = [*intrinsic_initial, 0]

    def fun(intrinsic_initial):
        ret = []
        fx, fy, cx, cy, val = intrinsic_initial
        intrinsic_mat = np.array(
                [
                    [fx, val, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ]
        )
        W_inv = intrinsic_mat @ intrinsic_mat.T
        intrinsic_inv = np.linalg.inv(intrinsic_mat)
        for i in fundamental.keys():
            for j in fundamental[i].keys():
                if i >= j:
                    continue
                u, s, v = np.linalg.svd(fundamental[i][j].T)
                u1, u2, u3 = u[:, 0, None], u[:, 1, None], u[:, 2, None]
                v1, v2, v3 = v[0, :, None], v[1, :, None], v[2, :, None]

                r1 = sorted(s)[2]
                r2 = sorted(s)[1]

                A = (r1 ** 2 * v1.T @ W_inv @ v1) @ np.linalg.pinv(u2.T @ W_inv @ u2)
                B = (r1 * r2 * v1.T @ W_inv @ v2) @ np.linalg.pinv(-u1.T @ W_inv @ u2)
                C = (r2 ** 2 * v2.T @ W_inv @ v2) @ np.linalg.pinv(u1.T @ W_inv @ u1)

                E1 = A - B
                E2 = B - C
                E3 = C - A

                ret.append(np.concatenate([E1, E2, E3]))

        return np.concatenate(ret).reshape(-1)

    ret = least_squares(fun, np.array(intrinsic_initial), method="lm", xtol=1e-10, ftol=1e-10)
    return ret.x

def daq(intrinsic_initial, fundamental, extrinsics):

    intrinsic_initial = intrinsic_initial.cpu().numpy()
    fx, fy, cx, cy = intrinsic_initial
    intrinsic_mat = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ]
        )
    intrinsic_initial = [*intrinsic_initial, 0, 0, 0, 0, 1]
    import sympy as sym

    def normplane():
        W_inv = intrinsic_mat @ intrinsic_mat.T
        x,y,z,XX = sym.symbols("x, y, z, xx", real=True)
        N = sym.Matrix([x, y, z])
        W_inv = sym.Matrix(W_inv)
        Q = sym.Matrix([[W_inv, W_inv @ N], [N.T @ W_inv, N.T @ W_inv @ N]])
        M = sym.Matrix(extrinsics[4])
        calib = M @ Q @ M.T
        a = sym.Eq(XX * W_inv[0, 0] - calib[0, 0], 0)
        b = sym.Eq(XX * W_inv[1, 1] - calib[1, 1], 0)
        c = sym.Eq(XX * W_inv[0, 2] - calib[0, 2], 0)
        d = sym.Eq(XX * W_inv[1, 2] - calib[1, 2], 0)

        ret = sym.solve([a, b, c, d], [x, y, z, XX])
        return np.array([ret[1][0], ret[1][1], ret[1][2]]).astype(np.float32)

    norm = normplane()
    homo_arr = []
    for i in fundamental.keys():
        for j in fundamental[i].keys():
            if i >= j:
                continue
            _, _, v = np.linalg.svd(fundamental[i][j].T)
            epi = v[-1]
            homo_arr.append(skew(epi) @ fundamental[i][j] + epi @ norm.T)
    
    homo_arr = np.array(homo_arr)

    def fun(intrinsic_initial):
        ret = []
        fx, fy, cx, cy, val1, val2, val3, val4, val5 = intrinsic_initial
        intrinsic_mat = np.array(
                [
                    [fx, val1, cx],
                    [val2, fy, cy],
                    [val3, val4, val5]
                ]
        )
        W_inv = intrinsic_mat @ intrinsic_mat.T

        for homo in homo_arr:
            DAQ = homo @ W_inv @ homo.T
            E = DAQ - W_inv
            ret.append(E)

        return np.concatenate(ret).reshape(-1)

    ret = least_squares(fun, np.array(intrinsic_initial),  method="lm", ftol=3e-16, xtol=3e-16)
    return (ret.x / ret.x[-1]).reshape(3, 3)


def preprocess_match(match_result):

    kps0_list = []
    kps1_list = []

    for (i, match) in enumerate(match_result):

        kps0 = match["kps0"]
        kps1 = match["kps1"]
        matches = match["matches"]

        kps0 = torch.stack([kps0[match_[0]] for match_ in matches])
        kps1 = torch.stack([kps1[match_[1]] for match_ in matches])

        kps0_list.append(kps0)
        kps1_list.append(kps1)

    return (torch.stack(kps0_list), torch.stack(kps1_list))

transform = tf.Compose(
    [
        tf.ToTensor(),
    ]
)

# Fundamental matrix from imgi to imgj. 
def get_fundamental_matrix(matching, imgi, imgj, args):

    imgi = transform(imgi).permute(1, 2, 0)
    imgj = transform(imgj).permute(1, 2, 0)
    result = runSuperGlueSinglePair(matching, imgi, imgj, 0, args)
    
    kps0_list, kps1_list = preprocess_match(result)
    kps0_list = kps0_list[0]
    kps1_list = kps1_list[0]

    kps0_list = cv2.UMat(kps0_list.cpu().numpy())
    kps1_list = cv2.UMat(kps1_list.cpu().numpy())

    F, mask = cv2.findFundamentalMat(kps0_list, kps1_list, cv2.FM_LMEDS)

    return F

def feasible_image_pair_candidates(extrinsics):

    pairs = {}
    num_images = len(extrinsics)
    for i in range(num_images):
        rot_mat_i = extrinsics[i][:3, :3]
        for j in range(i + 1, num_images):
            rot_mat_j = extrinsics[j][:3, :3]
            rot_mat_ij = torch.from_numpy(rot_mat_i @ np.linalg.inv(rot_mat_j))
            angle_rad = torch.acos((torch.trace(rot_mat_ij) - 1) / 2)
            angle_deg = angle_rad / np.pi * 180
            if torch.abs(angle_deg) < 30:
                i_entry = i
                j_entry = j

                if not i_entry in pairs.keys():
                    pairs[i_entry] = []
                if not j_entry in pairs.keys():
                    pairs[j_entry] = []
                pairs[i_entry].append(j_entry)
                pairs[j_entry].append(i_entry)

    return pairs

def data_mat(args): 

    ret = {}
    ckpt = torch.load(args.ckpt_path)

    json_prefix = "/".join(args.json_path.split("/")[:-1])

    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        },
        'superglue': {
                'weights': args.superglue_weight ,
                'sinkhorn_iterations': args.sinkhorn_iterations,
                'match_threshold': args.match_threshold,
        }
    }

    matching = Matching(config).eval().cuda()

    with open(args.json_path) as fp:
        json_file = json.load(fp)

    rot_matrices = [
        json_file["frames"][i]["transform_matrix"] \
        for i in range(len(json_file["frames"]))
        ] 

    # PPM
    rot_matrices = np.array(rot_matrices)
    num_img = rot_matrices.shape[0]

    img_dict = {}

    for i in range(num_img):
        img_path = os.path.join(json_prefix, json_file["frames"][i]["file_path"] + ".png")
        img_dict[i] = Image.open(img_path)

    feasible_pairs = feasible_image_pair_candidates(rot_matrices)
    
    fmatrix_ret = {}
    with torch.no_grad():
        for i in tqdm.tqdm(feasible_pairs.keys()):
            fmatrix_ret[i] = {}
            for j in feasible_pairs[i]:
                fmatrix = get_fundamental_matrix(matching, img_dict[i], img_dict[j], args)
                fmatrix_ret[i][j] = fmatrix

    intrinsic_initial = ckpt["camera_model"]["intrinsics_initial"]
    intrinsic_noise = ckpt["camera_model"]["intrinsics_noise"] 
    intrinsic_calibrated = intrinsic_initial + intrinsic_noise * intrinsic_initial
    
    projection_matrix = intrinsic_initial.cpu().numpy() @ rot_matrices

    mend = mendonca(intrinsic_initial, fmatrix_ret, projection_matrix)
    ck = classical_kruppa(intrinsic_initial, fmatrix_ret, projection_matrix)
    sp = simple_kruppa(intrinsic_initial, fmatrix_ret, projection_matrix)
    da = daq(intrinsic_initial, fmatrix_ret, projection_matrix)

    print(intrinsic_calibrated)
    print(mend)
    print(ck)
    print(sp)


if __name__ == "__main__":
    
    # Only works for blender dataset
 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path", 
        type=str,
        default="../data/nerf_synthetic/chair/transforms_train.json",
        help="path to json data json file"
    )
    parser.add_argument(
        "--mat_save_name",
        type=str,
        default="mat_result/chair_test.mat",
        help="matrix to save"        
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="170000.tar",
        help="path to ckpt"
    )

    parser.add_argument(
        "--superglue_weight",
        type=str,
        default="outdoor",
        help="Use outdoor weight or indoor weight",
    )
    parser.add_argument(
        "--max_keypoints", type=int, default=1024, help="Max number of keypoints"
    )
    parser.add_argument(
        "--nms_radius", type=int, default=4, help="NMS radius for SuperGlue"
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of sinkhorn iteration",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="Match threshold"
    )
    parser.add_argument(
        "--keypoint_threshold", type=float, default=0.005, help="Threshold of keypoints"
    )
    args = parser.parse_args()

    data_mat(args)