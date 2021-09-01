---
last_modified_at: 2021-08-31T22:28:00
classes: wide
title: "Self-Calibrating Neural Radiance Fields"
---

[Paper](https://arxiv.org/abs/2108.13826) | [Video](https://www.youtube.com/watch?v=_4u7p-cKnw0) | [Code](https://github.com/POSTECH-CVLab/SCNeRF)

## Qucik Intro

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/_4u7p-cKnw0" frameborder="0" allowfullscreen="true" width="600" height="400"> </iframe>
</figure>
</center>

## Overview

In this work, we propose a camera self-calibration algorithm for generic cameras with arbitrary non-linear distortions. We jointly learn the geometry of the scene and the accurate camera parameters without any calibration objects. Our camera model consists a pinhole model, radial distortion, and a generic noise model that can learn arbitrary non-linear camera distortions. While traditional self-calibration algorithms mostly rely on geometric constraints, we additionally incorporate photometric consistency. This requires learning the geometry of the scene and we use Neural Radiance Fields (NeRF). We also propose a new geometric loss function, viz., projected ray distance loss, to incorporate geometric consistency for complex non-linear camera models. We validate our approach on standard real image datasets and demonstrate our model can learn the camera intrinsics and extrinsics (pose) from scratch without COLMAP initialization. Also, we show that learning accurate camera models in differentiable manner allows us to improves PSNR over NeRF. We experimentally demonstrate that our proposed method is applicable to variants of NeRF. In addition, we use a set of images captured with a fish-eye lens to demonstrate that learning camera model jointly improves the performance significantly over the COLMAP initialization.

## Rendering without any camera information [[LLFF dataset]](https://github.com/Fyusion/LLFF)

Although our model does not adopt carefully calibrated camera information, i.e. COLMAP camera information, our model renders scenes clearly. 




<div style="float:left;">
    <div align="center">
        <iframe src="https://drive.google.com/file/d/1ml_3ucdnlRflkSBUSThjVTmgsJ7M6WNV/preview?start=1&loop=1" width="550" height="480" allow="autoplay" frameborder="0" allowfullscreen></iframe>
    </div>
</div>
<div style="float:right;">
    <div align="center">
        <iframe src="https://drive.google.com/file/d/1C6sP92idfi6Uzg7MWsuPPdQ54wQ3w4C3/preview?start=1&loop=1" width="550" height="480" allow="autoplay" frameborder="0" allowfullscreen></iframe>
    </div>
</div>


## Improvement over NeRF [[LLFF dataset]](https://github.com/Fyusion/LLFF)

Our algorithm improves NeRF in LLFF dataset. 

<div style="float:left;">
    <div align="center">
        <iframe src="https://drive.google.com/file/d/1458sv9hSC2ns1t4zo9C1Eoz-jGyRDmS-/preview?start=1&loop=1" width="550" height="480" allow="autoplay" frameborder="0" allowfullscreen></iframe>
    </div>
</div>
<div style="float:right;">
    <div align="center">
        <iframe src="https://drive.google.com/file/d/11W4P9oWZ6guumEu5E2WhF6-hXn1ghXg_/preview?start=1&loop=1" width="550" height="480" allow="autoplay" frameborder="0" allowfullscreen></iframe>
    </div>
</div>

## Improvement over NeRF++ [[Tanks and Temples dataset]](https://www.tanksandtemples.org/)

Our algorithm also improves NeRF++ in LLFF dataset. 

<div style="float:left;">
    <div align="center">
        <iframe src="https://drive.google.com/file/d/1lYTuBAJYITvVkLBMoRXIBWTNeWIT7Hi4/preview?start=1&loop=1" width="550" height="480" allow="autoplay" frameborder="0" allowfullscreen></iframe>
    </div>
</div>
<div style="float:right;">
    <div align="center">
        <iframe src="https://drive.google.com/file/d/1J_mDD-EMHSdAvxLFlA2l4dVL-5WKvuiO/preview?start=1&loop=1" width="550" height="480" allow="autoplay" frameborder="0" allowfullscreen></iframe>
    </div>
</div>


## Acknowledgement

We appreciate for all the ICCV reviewers for valuable comments. Their valuable suggestions have helped us to improve our paper.