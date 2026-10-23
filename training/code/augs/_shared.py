"""Shared helpers for augmentation ops (moved verbatim from the legacy
src/train/mri_augs.py monolith, 2026-08-23)."""
import math

import torch
import torch.nn.functional as F


def _rand(gen, lo, hi, shape=(1,), device="cpu"):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def _affine(x, angle=0.0, tx=0.0, ty=0.0, scale=1.0, shear=0.0):
    """Same affine for every slice of every study in the batch."""
    B, S, C, H, W = x.shape
    a = math.radians(angle)
    sh = math.radians(shear)
    cos, sin = math.cos(a), math.sin(a)
    # inverse-mapping affine matrix for grid_sample
    m = torch.tensor([[cos / scale, -sin / scale + math.tan(sh), tx],
                      [sin / scale, cos / scale, ty]],
                     dtype=x.dtype, device=x.device)
    flat = x.reshape(B * S, C, H, W)
    grid = F.affine_grid(m.unsqueeze(0).expand(B * S, -1, -1), flat.shape,
                         align_corners=False)
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    return out.reshape(B, S, C, H, W)
