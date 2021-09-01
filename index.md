---
permalink: /scnerf/
last_modified_at: 2021-08-31T22:28:00
layout: single
classes: wide
author_profile: true
title: "Self-Calibrating Neural Radiance Fields"
---

Paper Accepted in ICCV21


![](/assets/images/scnerf_teaser.png)


Author: [Yoonwoo Jeong](https://scholar.google.com/citations?user=HQ1PMggAAAAJ&hl=en&oi=ao), Seokjun Ahn, [Chris Choy](https://scholar.google.com/citations?user=2u8G5ksAAAAJ&hl=en&oi=ao), [Anima Anandkumar](https://scholar.google.com/citations?user=bEcLezcAAAAJ&hl=en&oi=ao), [Minsu Cho](https://scholar.google.com/citations?user=5TyoF5QAAAAJ&hl=en&oi=ao), [Jaesik Park](https://scholar.google.com/citations?user=_3q6KBIAAAAJ&hl=en&oi=ao)

[[Paper]](https://arxiv.org/abs/2108.13826) [[Code]](https://github.com/POSTECH-CVLab/SCNeRF) [[Video]](https://youtu.be/_4u7p-cKnw0)

## Qucik Intro

<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/_4u7p-cKnw0" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

## Abstract

In this work, we propose a camera self-calibration algorithm for generic cameras with arbitrary non-linear distortions. We jointly learn the geometry of the scene and the accurate camera parameters without any calibration objects. Our camera model consists of a pinhole model, a fourth order radial distortion, and a generic noise model that can learn arbitrary non-linear camera distortions. While traditional self-calibration algorithms mostly rely on geometric constraints, we additionally incorporate photometric consistency. This requires learning the geometry of the scene, and we use Neural Radiance Fields (NeRF). We also propose a new geometric loss function, viz., projected ray distance loss, to incorporate geometric consistency for complex non-linear camera models. We validate our approach on standard real image datasets and demonstrate that our model can learn the camera intrinsics and extrinsics (pose) from scratch without COLMAP initialization. Also, we show that learning accurate camera models in a differentiable manner allows us to improve PSNR over baselines. 
Our module is an easy-to-use plugin that can be applied to NeRF variants to improve performance. The code and data are currently available at [[link]](https://github.com/POSTECH-CVLab/SCNeRF).

## Method

We provide descriptions of our algorithm in the following link: [[link]](/scnerf_contents)


## Concurrent Works
We compare typical concurrent works with ours. 

1. NeRF--: Neural Radiance Fields Without Known Camera Parameters
    - [[Paper]](https://arxiv.org/abs/2102.07064) [[Code]](https://github.com/ActiveVisionLab/nerfmm)
    - This paper has proposed a self-calibration algorithm that enables rendering without camera information prior. By jointly optimizing NeRF parameters and camera parameters, NeRF-- successfully render images in several datasets. 
    - Our paper additionally propose camera parameters that reflect non-linear distortions, and the projected ray distance loss that improves geometric consistency of rays that are generated from correspondences.

2. BARF: Bundle-Adjusting Neural Radiance Fields
    - [[Paper]](https://arxiv.org/abs/2104.06405) [[Code]](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)


## Qualitative Results

### Rendering without any camera information [[LLFF dataset]](https://github.com/Fyusion/LLFF)

Although our model does not adopt carefully calibrated camera information, i.e. COLMAP camera information, our model renders scenes clearly. 

|-|-| -|-|-| -|-|-| -|-|-| -|
| <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | <figure class="video_container"><iframe src="https://drive.google.com/file/d/15_u7XEh2ondFeUJHlp6GsexHvTpaXn3s/preview" frameborder="0" allowfullscreen="true"> </iframe> | 


### Improvement over NeRF++ [[Tanks and Temples dataset]](https://www.tanksandtemples.org/)

 |-|-| -|-|-|
| <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/m60_ours.mp4" type="video/mp4"></video></figure> |  <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/playground_ours.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/train_ours.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/truck_ours.mp4" type="video/mp4"></video></figure> | 
