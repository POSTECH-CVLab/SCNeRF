import torch
import tqdm
import torch.nn as nn
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, "..")

from thirdparty.superglue.models.matching import Matching

# For debugging

def image_pair_candidates(extrinsics, args, i_map=None):

    # i_map is used when provided extrinsics are not having sequentiall 
    # index. i_map is a list of ints where each element corresponds to 
    # image index. 
    
    pairs = {}

    assert i_map is None or len(i_map) == len(extrinsics)

    num_images = len(extrinsics)
    
    for i in range(num_images):
        
        rot_mat_i = extrinsics[i][:3, :3]
        
        for j in range(i + 1, num_images):
            
            rot_mat_j = extrinsics[j][:3, :3]
            rot_mat_ij = torch.from_numpy(rot_mat_i @ np.linalg.inv(rot_mat_j))
            angle_rad = torch.acos((torch.trace(rot_mat_ij) - 1) / 2)
            angle_deg = angle_rad / np.pi * 180
            
            if torch.abs(angle_deg) < args.pairing_angle_threshold:
                
                i_entry = i if i_map is None else i_map[i]
                j_entry = j if i_map is None else i_map[j]

                if not i_entry in pairs.keys():
                    pairs[i_entry] = []
                if not j_entry in pairs.keys():
                    pairs[j_entry] = []
                    
                pairs[i_entry].append(j_entry)
                pairs[j_entry].append(i_entry)

    return pairs

def init_superglue(args, rank):
    config = {
        "superpoint": {
            "nms_radius": args.nms_radius,
            "keypoint_threshold": args.keypoint_threshold,
            "max_keypoints": args.max_keypoints,
        },
        "superglue": {
            "weights": args.superglue_weight \
                    if hasattr(args, "superglue_weight") else \
                    args.weight,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "match_threshold": args.match_threshold,
        },
    }
    superglue = Matching(config).eval().to(rank)
    return superglue

def runSIFTSinglePair(sift , img0, img1, rank, args):

    if isinstance(img0, torch.Tensor):
        img0 = img0.cpu().numpy()
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    
    # Convert RGB images to gray images 
    img0_gray = (
        0.2989 * (img0[:, :, 0])
        + 0.5870 * (img0[:, :, 1])
        + 0.1140 * (img0[:, :, 2])
    )
    
    img1_gray = (
        0.2989 * (img1[:, :, 0])
        + 0.5870 * (img1[:, :, 1])
        + 0.1140 * (img1[:, :, 2])
    )

    kp0, des0 = sift.detectAndCompute((img0_gray * 255).astype(np.uint8),None)
    kp1, des1 = sift.detectAndCompute((img1_gray * 255).astype(np.uint8),None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des0,des1,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])


    kp0 = torch.from_numpy(np.array([kp.pt for kp in kp0])).to(rank)
    kp1 = torch.from_numpy(np.array([kp.pt for kp in kp1])).to(rank)

    matches = torch.from_numpy(np.array([[match[0].queryIdx, match[0].trainIdx] for match in good])).to(rank)

    return  [
        {
            "kps0": kp0.detach(),
            "kps1": kp1.detach(),
            "matches": matches.detach()
        }
    ]


def runSuperGlueSinglePair(superglue, img0, img1, rank, args):

    if isinstance(img0, np.ndarray):
        img0 = torch.from_numpy(img0)
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)

    # Must have 3 channels
    assert img0.shape[2] == 3 and img1.shape[2] == 3
    
    # Convert RGB images to gray images 
    img0_gray = (
        0.2989 * (img0[:, :, 0])
        + 0.5870 * (img0[:, :, 1])
        + 0.1140 * (img0[:, :, 2])
    ).to(rank)
    
    img1_gray = (
        0.2989 * (img1[:, :, 0])
        + 0.5870 * (img1[:, :, 1])
        + 0.1140 * (img1[:, :, 2])
    ).to(rank)

    pred = superglue(
        {
            "image0": img0_gray[None, None, :, :],
            "image1": img1_gray[None, None, :, :]
        }
    )

    pred = {k: v[0] for k, v in pred.items()}

    match_src = torch.where(pred["matches0"] != -1)[0]
    match_trg = pred["matches0"][match_src]
    kps0, kps1 = pred["keypoints0"], pred["keypoints1"]

    matches = torch.stack([match_src, match_trg], dim=1)
    conf = pred['matching_scores0'][match_src]
    return [
        {
            "kps0": kps0.detach(),
            "kps1": kps1.detach(),
            "matches": matches.detach(),
            "conf": conf
        }
    ]


def runSuperGlue(superglue, img_pairs, match_num, rank):

    ret = []
    with torch.no_grad():
        for i in range(len(img_pairs)):
            src_gray = (
                0.2989 * (img_pairs[i][0][:, :, 0])
                + 0.5870 * (img_pairs[i][0][:, :, 1])
                + 0.1140 * (img_pairs[i][0][:, :, 2])
            ).to(rank)
            trg_gray = (
                0.2989 * (img_pairs[i][1][:, :, 0])
                + 0.5870 * (img_pairs[i][1][:, :, 1])
                + 0.1140 * (img_pairs[i][1][:, :, 2])
            ).to(rank)
            pred = superglue(
                {
                    "image0": src_gray[None, None, :, :],
                    "image1": trg_gray[None, None, :, :],
                }
            )
            pred = {k: v[0] for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches0, conf0 = pred["matches0"], pred["matching_scores0"]
            matched_src = torch.argsort(conf0, descending=True)[
                : match_num][: len(torch.where(matches0 != -1)[0])]
            matched_trg = matches0[matched_src]
            conf = conf0[matched_src]
            matches = torch.stack([matched_src, matched_trg], dim=1)
            ret.append(
                {
                    "kps0": kpts0.detach(),
                    "kps1": kpts1.detach(),
                    "desc1": pred["descriptors0"].detach(),
                    "desc2": pred["descriptors1"].detach(),
                    "matches": matches.detach(),
                    "conf": conf.detach(),
                }
            )

    return ret
