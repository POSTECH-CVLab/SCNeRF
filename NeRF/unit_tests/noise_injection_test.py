import numpy as np
import torch
from .utils import print_separator, R_to_axis_angle, tol

def unit_test_noise_injection(**kwargs):
    
    # This test checks whether the noise is injected properly 
    
    print_separator()
    msg = "Failed to pass the unit test named noise_injection_test"
    print("Starting Unit Test : noise_injection_test")
    
    # Basic Argument Existence Test
    assert "args" in kwargs.keys(), msg
    assert "i_train" in kwargs.keys(), msg
    assert "i_val" in kwargs.keys(), msg
    assert "i_test" in kwargs.keys(), msg
    assert "gt_intrinsic" in kwargs.keys(), msg
    assert "gt_extrinsic" in kwargs.keys(), msg
    assert "hwf" in kwargs.keys(), msg
    assert "noisy_extrinsic" in kwargs.keys(), msg
    
    args = kwargs["args"]
    i_train, i_val, i_test = (
        kwargs["i_train"], kwargs["i_val"], kwargs["i_test"]
    )
    gt_intrinsic = kwargs["gt_intrinsic"].detach().cpu()
    gt_focal = gt_intrinsic[0][0]
    gt_extrinsic = kwargs["gt_extrinsic"].detach().cpu()
    (H, W, noisy_focal) = kwargs["hwf"]
    noisy_extrinsic = torch.from_numpy(kwargs["noisy_extrinsic"])
    
    # Basic Argument Option Test
    assert hasattr(args, "debug") and args.debug, msg
    assert hasattr(args, "initial_noise_size_intrinsic"), msg
    
    # No overlapping views between train<->val and train<->test.
    for train_idx in i_train:
        assert not train_idx in i_test, msg
        assert not train_idx in i_val, msg
        
    for val_idx in i_val:
        assert not val_idx in i_train, msg
    
    for test_idx in i_test:
        assert not test_idx in i_train, msg
    
    # Test and validation poses must be the same with the GT poses.
    assert gt_extrinsic.shape == noisy_extrinsic.shape, msg
    assert (
        gt_extrinsic[i_test] - noisy_extrinsic[i_test]
    ).abs().max() < tol, msg
    assert (
        gt_extrinsic[i_val] - noisy_extrinsic[i_val]
    ).abs().max() < tol, msg
    
    # The difference between noisy focal length and the GT focal length
    # must be the same as the value in the argument.
    if args.run_without_colmap:
        pass
    else:
        noise_size = (noisy_focal - gt_focal) / gt_focal
        assert torch.abs(noise_size - args.initial_noise_size_intrinsic) < tol, msg

        # All the noisy poses must be located within the angle in the argument.
        rot_diff = (
            torch.inverse(gt_extrinsic[:, :3, :3]) @ noisy_extrinsic[:, :3, :3]
        )
        rot_diff_detached = rot_diff.detach().cpu().numpy()
        rot_diff_angle = torch.tensor(
            [
                R_to_axis_angle(rot_mat)[1] for rot_mat in rot_diff_detached
            ]
        )
        rot_diff_angle_rad = (rot_diff_angle / np.pi * 180).abs()
        assert torch.all(
            rot_diff_angle_rad <= args.initial_noise_size_rotation + tol
        ), msg
    
    print("Passed Unit Test : noise_injection_test")
    print_separator()
