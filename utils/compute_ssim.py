import torch
from skimage.metrics import structural_similarity
import numpy as np

@torch.no_grad()
def compute_ssim(ground_truth, predicted, full=True):
    # The arguments to `structural_similarity` have been chosen to match
    # PixelSplat (apart from `full = full`)
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
            full=full,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    if full:
        ssim = [spatial for _, spatial in ssim]
    ssim = np.array(ssim)
    ssim = torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
    assert not torch.isnan(ssim).any(), "SSIM has NaNs"
    return ssim
