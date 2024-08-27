<div align="center">

# Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs

[**Brandon Smart**](https://scholar.google.com/citations?user=k_jn6-EAAAAJ)<sup>1</sup> · [**Chuanxia Zheng**](https://chuanxiaz.com/)<sup>2</sup> · [**Iro Laina**](https://scholar.google.com/citations?user=n9nXAPcAAAAJ)<sup>2</sup> · [**Victor Adrian Prisacariu**](https://www.robots.ox.ac.uk/~victor/)<sup>1</sup> 

<sup>1</sup>Active Vision Lab · <sup>2</sup>Visual Geometry Group

University of Oxford

<a href='https://splatt3r.active.vision'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2408.13912'><img src='https://img.shields.io/badge/arXiv Paper-red'></a>
<a href='https://huggingface.co/spaces/brandonsmart/splatt3r'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

</div>

![Teaser for Splatt3R](assets/overview.svg)

Official implementation of `Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs`, a feed-forward model that can directly predict 3D Gaussians from uncalibrated images.

## News

- [2024/08/27] 🔥 We release the initial version of the codebase, the paper, the project webpage, and the Gradio demo!!

## Installation

1. Clone Splatt3R  
```bash
git clone https://github.com/btsmart/splatt3r.git
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

The Gradio demo can be run using `python demo.py`, which loads our trained checkpoint from Hugging Face.

This demo generates a `.ply` file that represents the scene, which can be downloaded and rendered using online 3D Gaussian Splatting viewers such as [here](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php?art=1&cu=0,-1,0&cp=0,1,0&cla=1,0,0&aa=false&2d=false&sh=0) or [here](https://playcanvas.com/supersplat/editor).

Our example images and `.ply` files are available for download [here](https://huggingface.co/brandonsmart/splatt3r_v1.0/tree/main/demo_examples).

## Training

Our training run can be recreated by running `python main.py configs/main.yaml`. Other configurations, such as those for the ablations, can be found in the `configs` folder.

## BibTeX

If you find Splatt3R useful for your research and applications, please cite us using this BibTex:
```
@article{smart2024splatt3r,
      title={Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs}, 
      author={Brandon Smart and Chuanxia Zheng and Iro Laina and Victor Adrian Prisacariu},
      year={2024},
      eprint={2408.13912},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13912}, 
}
```

## License
 [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
