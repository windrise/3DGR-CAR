<div align="center">
<h1>3DGR-CAR: Coronary artery reconstruction from ultra-sparse 2D X-ray views with a 3D Gaussians representation</h1>
<h3> Accepted at MICCAI 2024 </h3>

Authors: Xueming Fu, Yingtai Li, Fenghe Tang, Jun Li, Mingyue Zhao, Gao-Jun Teng and S. Kevin Zhou

<!--
[![arxiv paper](https://img.shields.io/badge/arxiv-paper-orange)](https://github.com/windrise/3DGR-CAR/tree/main)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![authors](https://img.shields.io/badge/by-windrise-green)](https://github.com/windrise)
-->

</div>

<p align="center">
<img src="Pic/Result.png" width="70%">
</p>


## Abstract
Reconstructing 3D coronary arteries is important for coronary artery disease diagnosis, treatment planning and operation navigation. Traditional reconstruction techniques often require many projections, while reconstruction from sparse-view X-ray projections is a potential way of reducing radiation dose. However, the extreme sparsity of coronary arteries in a 3D volume and ultra-limited number of projections pose significant challenges for efficient and accurate 3D reconstruction. To this end, we propose 3DGR-CAR, a 3D Gaussian Representation for Coronary Artery Reconstruction from ultra-sparse X-ray projections. We leverage 3D Gaussian representation to avoid the inefficiency caused by the extreme sparsity of coronary artery data and propose a Gaussian center predictor to overcome the noisy Gaussian initialization from ultra-sparse view projections. The proposed scheme enables fast and accurate 3D coronary artery reconstruction with only 2 views. Experimental results on two datasets indicate that the proposed approach significantly outperforms other methods in terms of voxel accuracy and visual quality of coronary arteries.

<p align="center">
<img src="Pic/framework.png" width="70%">
</p>

## Introduction
**Is it possible to utilize a really sparse number of 2D X-ray views to reconstruct coronary arteries in 3D?**
<p align="center">
<img src="Pic/intro.png" width="70%">
</p>

## Installation

```
  # it is recommanded to use conda
  conda create -n 3dgs-car python=3.9
  conda activate 3dgs-car
  
  # install dependencies
  pip install -r requirements.txt
  
  # gaussian splatting 
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
  pip install ./diff-gaussian-rasterization
  
  # simple-knn
  pip install ./simple-knn
  
```

## Dataset
We use dataset from .[ASOCA](https://asoca.grand-challenge.org/) and .[ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT). 




## 🤝Acknowledgement

Our repo is built upon .[Gasussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) [Splat image](https://github.com/szymanowiczs/splatter-image) and [NeRP](https://github.com/liyues/NeRP). Thanks to their work.

<!--
## Citation
```

```
-->
