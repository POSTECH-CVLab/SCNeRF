import os

from random import random
import numpy as np
import torch
import torchvision.transforms as TF

import matplotlib.pyplot as plt

from .utils import tol, print_separator

match_num = 50

def unit_test_matches(**kwargs):
    
    print_separator()
    msg = "Failed to pass the unit test named matches"
    print("Starting Unit Test : matches")
    
    dirname = "_unit_test_matches_result"
    
    # Check whether argument is currently provided. 
    assert "args" in kwargs.keys(), msg
    assert "result" in kwargs.keys(), msg 
    assert "img_i" in kwargs.keys(), msg
    assert "img_j" in kwargs.keys(), msg
    assert "img_i_idx" in kwargs.keys(), msg
    assert "img_j_idx" in kwargs.keys(), msg
    
    args= kwargs["args"]
    result = kwargs["result"]
    img_i, img_j = kwargs["img_i"], kwargs["img_j"]
    img_i_idx, img_j_idx = kwargs["img_i_idx"], kwargs["img_j_idx"]
    kps1, kps2 = result
    W = img_i.shape[1]
    
    # Draw matches and save them
    assert hasattr(args, "datadir"), msg    
    scene_name = args.datadir.split("/")[-1]
    scene_path = os.path.join(dirname, scene_name)
    os.makedirs(scene_path, exist_ok=True)
    img_name = "{}_{}.png".format(img_i_idx, img_j_idx)
    img_path = os.path.join(scene_path, img_name)
    
    img_cat = torch.cat([img_i, img_j], dim=1)
    img_cat_pil = TF.ToPILImage()(img_cat.permute(2, 0, 1))
    plt.imshow(img_cat_pil)
    
    i_visualize = np.random.choice(range(len(kps1[0])), match_num)
    
    for i in i_visualize: 
        kp1, kp2 = kps1[i].cpu().numpy(), kps2[i].cpu().numpy()
        color = (random(), random(), random())
        plt.plot([kp1[0], kp2[0]+W], [kp1[1], kp2[1]], c=color, lw=2)
        
    plt.savefig(img_path)    
    plt.close()
    print_separator() 