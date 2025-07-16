#!/usr/bin/env python3
import argparse, os, sys, torch
from huggingface_hub import hf_hub_download

sys.path.extend(['src/mast3r_src', 'src/mast3r_src/dust3r', 'src/pixelsplat_src'])

from dust3r.utils.image import load_images          
import main                                         
import utils.export as export                       

parser = argparse.ArgumentParser()
parser.add_argument('img1', help='first RGB image')
parser.add_argument('img2', help='second RGB image')
parser.add_argument('--outdir', default='results', help='output folder')
parser.add_argument('--model-dir', default=None,
                    help='where to cache / load the Splatt3R checkpoint')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Locate (or download once) the official checkpoint ─────────────────────────
ckpt = hf_hub_download(
    repo_id             = "brandonsmart/splatt3r_v1.0",
    filename            = "epoch=19-step=1200.ckpt",
    local_dir           = args.model_dir,          
    local_dir_use_symlinks = False,
    resume_download     = True,
)

# ── Build model & run inference ───────────────────────────────────────────────
model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device)
model.eval().to(device)

# 512×512 centre-crop & normalisation (same helper as demo)
imgs = load_images([args.img1, args.img2], size=512, verbose=False)
for im in imgs:
    im['img']          = im['img'].to(device)
    im['original_img'] = im['original_img'].to(device)
    im['true_shape']   = torch.as_tensor(im['true_shape'])

# forward pass → two Gaussian maps
pred1, pred2 = model(imgs[0], imgs[1])

os.makedirs(args.outdir, exist_ok=True)
ply_path = os.path.join(args.outdir, 'gaussians.ply')
export.save_as_ply(pred1, pred2, ply_path)
print(f"Wrote {ply_path}")
