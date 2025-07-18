<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)

<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>

## Unofficial (fork) implementation of LatentSync
[original repo](https://github.com/bytedance/LatentSync/)

Optimizations were applied to with `.compile()` and version updates for Pytroch and other cuda-based libs.
Using Pytorch profiler. The full execution (CUDA) time improved from 420 secs to 397 secs.

A PoC implementation of optical flow was added to the loss function. This is computed per video and it was inspired from: [VideoJAM](https://hila-chefer.github.io/videojam-paper.github.io/VideoJAM_arxiv.pdf)

# Download dataset

To download the HDTF data. Follow instruction [link](https://github.com/universome/HDTF)

```bash
python download.py --output_dir path-to-dataset/
```

# Run

```bash
./train_unet_test.sh
```
