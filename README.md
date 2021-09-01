# Self-Calibrating Neural Radiance Fields, ICCV, 2021

[Project Page](jeongyw12382.github.io/scnerf) | [Paper](https://arxiv.org/abs/2108.13826) | [Video](https://www.youtube.com/watch?v=_4u7p-cKnw0)

<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/_4u7p-cKnw0" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


## Overview

In this work, we propose a camera self-calibration algorithm for generic cameras with arbitrary non-linear distortions. We jointly learn the geometry of the scene and the accurate camera parameters without any calibration objects. Our camera model consists a pinhole model, radial distortion, and a generic noise model that can learn arbitrary non-linear camera distortions. While traditional self-calibration algorithms mostly rely on geometric constraints, we additionally incorporate photometric consistency. This requires learning the geometry of the scene and we use Neural Radiance Fields (NeRF).
We also propose a new geometric loss function, viz., projected ray distance loss, to incorporate geometric consistency for complex non-linear camera models. We validate our approach on standard real image datasets and demonstrate our model can learn the camera intrinsics and extrinsics (pose) from scratch without COLMAP initialization. Also, we show that learning accurate camera models in differentiable manner allows us to improves PSNR over NeRF. We experimentally demonstrate that our proposed method is applicable to variants of NeRF. In addition, we use a set of images captured with a fish-eye lens to demonstrate that learning camera model jointly improves the performance significantly over the COLMAP initialization.

# News 
- 2021-08-16: The first version of Self-Calibrating Neural Radiance Fields is published

# Pre-requisite

## Requirements
- Ubuntu 16.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.7 or higher
- Hardware Spec
    - GPUs 11GB (2080ti) or larger capacity
    - For NeRF++, 2GPUs(2080ti) are required to reproduce the result
    - For FishEyeNeRF experiments, we have used 4GPUs(V100). 

## Installation
- We recommed to use conda for installation. All the requirements for two codes, NeRF and NeRF++, are included in the requirements.txt

    ```
    conda create -n icn python=3.8
    conda activate icn
    pip install -r requirements.txt
    git submodule update --init --recursive
    ```


# Pretrained Weights & Qualitative Results
Here, we provide pretrained weights for users to easily reproduce results in the paper. You can download the pretrained weight in the following link. In the link, we provide all the weights of experiments, reported in our paper. To load the pretrained weight, add the following argument at the end of argument in each script. In the zip file, we have also included qualitative results that are used in our paper. 

Link to download the pretrained weight: [[link]](https://drive.google.com/file/d/1rgJ6CpJh9EzmtOB7Q_mp6N4vbNRpE9Gs/view?usp=sharing)

# Raw-data
We use three datasets for evaluation: LLFF dataset, tanks and temples dataset, and FishEyeNeRF dataset (Images captured with a fish-eye lens). 
- LLFF dataset: [[link]](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- Tanks and Temples dataset: [[link]](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view?usp=sharing)
- FishEyeNeRF: [[link]](https://drive.google.com/file/d/1VhnpMUIKEak4TBpKY4H8vMJJe--FM2k5/view?usp=sharing)

Put the data in the directory "data/" then add soft link with one of the following:
```python3
ln -s data/nerf_llff_data NeRF/data
ln -s data/tanks_and_temples nerfplusplus/data
ln -s data/FishEyeNeRF nerfplusplus/data/fisheyenerf
```

# Demo Code

The demo code is available at "demo.sh" file. This code runs curriculum learning in NeRF architecture. Please install the aformentioned requirements before running the code. To run the demo code, run the follwing script:

```python3
sh demo.sh
```

If you want to reproduce the results that are reported in our main paper, run the scripts in the "scripts" directory.

```
Main Table 1: Self-Calibration Experiment (LLFF)
Main Table 2: Improvement over NeRF (LLFF)
Main Table 3: Improvement over NeRF++ (Tanks and Temples)
Main Table 4: Improvement over NeRF++ (Images with a fish-eye lens)
```

Code Example: 
```python3
sh scripts/main_table_1/fern/main1_fern_ours.sh
sh scripts/main_table_2/fern/main2_fern_ours.sh
sh scripts/main_table_3/main_3_m60.sh
sh scripts/main_table_4/globe_ours.sh
```

# Citing Self-Calibrating Neural Radiance Fields
    @inproceedings{SCNeRF2021,
        author = {Yoonwoo Jeong, Seokjun Ahn, Christopehr Choy, Animashree Anandkumar, 
        Minsu Cho, and Jaesik Park},
        title = {Self-Calibrating Neural Radiance Fields},
        booktitle = {ICCV},
        year = {2021},
    }

# Acknowledgements

We appreciate for all the ICCV reviewers for valuable comments. Their valuable suggestions have helped us to improve our paper. 