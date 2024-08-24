# Splatt3R: Zero-shot Gaussian Splatting from Uncalibarated Image Pairs

![Teaser for Splatt3R](assets/overview.svg)

Official implementation of `Zero-shot Gaussian Splatting from Uncalibarated Image Pairs`

[Project Page](https://btsmart.github.io/splatt3r/index.html), [Splatt3R arXiv](https://arxiv.org/), [Demo Page](https://huggingface.co/spaces/brandonsmart/splatt3r)

## Installation

1. Clone Splatt3R  
```bash
git clone <redacted github link>
cd splatt3r
```

2. Setup Anaconda Environment
```bash
conda env create -f environment.yml
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

3. (Optional) Compile the CUDA kernels for RoPE (as in MASt3R and CroCo v2)

```bash
cd src/dust3r_src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../
```

## Checkpoints

We train our model using the pretrained `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` checkpoint from the MASt3R authors, available from [the MASt3R GitHub repo](https://github.com/naver/mast3r). This checkpoint is placed at the file path `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`.

A pretrained Splatt3R model can be downloaded [here](https://huggingface.co/brandonsmart/splatt3r_v1.0/blob/main/epoch%3D19-step%3D1200.ckpt).

## Data

We use ScanNet++ to train our model. We download the data from the [official ScanNet++ homepage](https://kaldir.vc.in.tum.de/scannetpp/) and process the data using SplaTAM's modified version of [the ScanNet++ toolkit](https://github.com/Nik-V9/scannetpp). We save the processed data to the 'processed' subfolder of the ScanNet++ root directory.

Our generated test coverage files, and our training and testing splits, can be downloaded [here](https://huggingface.co/brandonsmart/splatt3r_v1.0/tree/main/scannetpp), and placed in `data/scannetpp`.

## Demo

The Gradio demo can be run using `python demo.py <checkpoint_path>`, replacing `<checkpoint_path>` with the trained network path. A checkpoint will be available for the public release of this code.

This demo generates a `.ply` file that represents the scene, which can be downloaded and rendered using online 3D Gaussian Splatting viewers such as [here](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php?art=1&cu=0,-1,0&cp=0,1,0&cla=1,0,0&aa=false&2d=false&sh=0) or [here](https://playcanvas.com/supersplat/editor).

Our example images and `.ply` files are available for download [here](https://huggingface.co/brandonsmart/splatt3r_v1.0/tree/main/demo_examples).

## Training

Our training run can be recreated by running `python main.py configs/main.yaml`. Other configurations, such as those for the ablations, can be found in the `configs` folder.

## BibTeX

Forthcoming arXiv citation