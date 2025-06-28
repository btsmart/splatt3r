#!/usr/bin/env python3
# minimal, headless Splatt3R inference on one stereo pair
import argparse, os, sys, torch
from huggingface_hub import hf_hub_download

# add vendored libs to PYTHONPATH
sys.path.extend(['src/mast3r_src', 'src/mast3r_src/dust3r', 'src/pixelsplat_src'])

from dust3r.utils.image import load_images                 # identical to demo.py :contentReference[oaicite:1]{index=1}
import main                                                # provides MAST3RGaussians
import utils.export as export                              # save_as_ply() :contentReference[oaicite:2]{index=2}

parser = argparse.ArgumentParser()
parser.add_argument('img1')
parser.add_argument('img2')
parser.add_argument('--outdir', default='results')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download / locate the official checkpoint (≈ 1.3 GB)
ckpt = hf_hub_download(
    repo_id="brandonsmart/splatt3r_v1.0",
    filename="epoch=19-step=1200.ckpt",
    local_dir="/cluster/51/dengnick/models/splatt3r",
    local_dir_use_symlinks=False,
    resume_download=True,    # handy if the job is pre-emptible
)

model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device)
model.eval().to(device)

# 512×512 centre-crop & normalisation (same helper as demo)
imgs = load_images([args.img1, args.img2], size=512, verbose=False)
for im in imgs:                                  # move tensors onto GPU
    im['img'] = im['img'].to(device)
    im['original_img'] = im['original_img'].to(device)
    im['true_shape'] = torch.as_tensor(im['true_shape'])

# forward pass → two Gaussian maps
pred1, pred2 = model(imgs[0], imgs[1])

os.makedirs(args.outdir, exist_ok=True)
ply_path = os.path.join(args.outdir, 'gaussians.ply')
export.save_as_ply(pred1, pred2, ply_path)       # write ~40 MB PLY
print(f"✨ wrote {ply_path}")
