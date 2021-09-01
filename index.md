---
last_modified_at: 2021-08-31T22:28:00
classes: wide
author_profile: true
title: "Self-Calibrating Neural Radiance Fields"
---


## Qucik Intro

<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/_4u7p-cKnw0" frameborder="0" allowfullscreen="true" width="600" height="400"> </iframe>
</figure>

## Abstract

In this work, we propose a camera self-calibration algorithm for generic cameras with arbitrary non-linear distortions. We jointly learn the geometry of the scene and the accurate camera parameters without any calibration objects. Our camera model consists of a pinhole model, a fourth order radial distortion, and a generic noise model that can learn arbitrary non-linear camera distortions. While traditional self-calibration algorithms mostly rely on geometric constraints, we additionally incorporate photometric consistency. This requires learning the geometry of the scene, and we use Neural Radiance Fields (NeRF). We also propose a new geometric loss function, viz., projected ray distance loss, to incorporate geometric consistency for complex non-linear camera models. We validate our approach on standard real image datasets and demonstrate that our model can learn the camera intrinsics and extrinsics (pose) from scratch without COLMAP initialization. Also, we show that learning accurate camera models in a differentiable manner allows us to improve PSNR over baselines. 
Our module is an easy-to-use plugin that can be applied to NeRF variants to improve performance. The code and data are currently available at [[link]](https://github.com/POSTECH-CVLab/SCNeRF).


## Rendering without any camera information [[LLFF dataset]](https://github.com/Fyusion/LLFF)

Although our model does not adopt carefully calibrated camera information, i.e. COLMAP camera information, our model renders scenes clearly. 

<div class="table-wrapper" markdown="block">

|-|-|-|
| <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/fern.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/flower.mp4" type="video/mp4"></video></figure>  | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/fortress.mp4" type="video/mp4"></video></figure>| <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/horns.mp4" type="video/mp4"></video></figure>| <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/leaves.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/orchids.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/room.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/trex.mp4" type="video/mp4"></video></figure> |
</div>



## Improvement over NeRF++ [[Tanks and Temples dataset]](https://www.tanksandtemples.org/)

<div class="table-wrapper" markdown="block">
 |-|-| -|-|-|
| <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/m60_ours.mp4" type="video/mp4"></video></figure> |  <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/playground_ours.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/train_ours.mp4" type="video/mp4"></video></figure> | <figure class="video_container"><video autoplay="" loop="" controls width="400"><source src="/assets/videos/truck_ours.mp4" type="video/mp4"></video></figure> | 
</div>
