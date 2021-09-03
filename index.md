---
last_modified_at: 2021-08-31T22:28:00
classes: wide
title: "Self-Calibrating Neural Radiance Fields"
---

[Paper](https://arxiv.org/abs/2108.13826) | [Video](https://www.youtube.com/embed/wsjx6geduvk) | [Code](https://github.com/POSTECH-CVLab/SCNeRF)

## Quick Intro

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/wsjx6geduvk" frameborder="0" allowfullscreen="true" width="100%" style="min-height: 450px;"> </iframe>
</figure>
</center>

## Overview

In this work, we propose a camera self-calibration algorithm for generic cameras with arbitrary non-linear distortions. We jointly learn the geometry of the scene and the accurate camera parameters without any calibration objects. Our camera model consists a pinhole model, radial distortion, and a generic noise model that can learn arbitrary non-linear camera distortions. While traditional self-calibration algorithms mostly rely on geometric constraints, we additionally incorporate photometric consistency. This requires learning the geometry of the scene and we use Neural Radiance Fields (NeRF). We also propose a new geometric loss function, viz., projected ray distance loss, to incorporate geometric consistency for complex non-linear camera models. We validate our approach on standard real image datasets and demonstrate our model can learn the camera intrinsics and extrinsics (pose) from scratch without COLMAP initialization. Also, we show that learning accurate camera models in differentiable manner allows us to improves PSNR over NeRF. We experimentally demonstrate that our proposed method is applicable to variants of NeRF. In addition, we use a set of images captured with a fish-eye lens to demonstrate that learning camera model jointly improves the performance significantly over the COLMAP initialization.

## Rendered Images with Self-Calibrating NeRF [[LLFF dataset]](https://github.com/Fyusion/LLFF)
#### (no camera information is provided)

Although our model does not adopt carefully calibrated camera information, i.e. COLMAP camera information, our model renders scenes clearly. 

{% include nerf_wo_colmap.html height="50" unit="%" duration="7" %}

## Improvement over NeRF [[LLFF dataset]](https://github.com/Fyusion/LLFF)

Our algorithm improves NeRF in LLFF dataset. 

{% include nerf_w_colmap.html height="50" unit="%" duration="7" %}

## Improvement over NeRF++ [[Tanks and Temples dataset]](https://www.tanksandtemples.org/)

Our algorithm also improves NeRF++ in Tanks and Temples dataset. 

{% include npp_w_colmap.html height="50" unit="%" duration="7" %}

## Acknowledgement

We appreciate for all the ICCV reviewers for valuable comments. Their valuable suggestions have helped us to improve our paper.
