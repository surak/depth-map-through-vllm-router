"""Run VGGT or VGGT-Omega on a list of image files."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .mapping import Reconstruction


def unproject(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """World points from depth (S, H, W[, 1]) and world-to-camera extrinsics (S, 3, 4)."""
    if depth.ndim == 4:
        depth = depth[..., 0]
    s, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation, translation = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    return np.einsum("sji,shwj->shwi", rotation, cam - translation[:, None, None, :]).astype(np.float32)


class VGGTRunner:
    def __init__(self, path: str, device: str = "cuda"):
        import torch
        from vggt.models.vggt import VGGT

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = VGGT.from_pretrained(path).to(device).eval()
        self.lock = threading.Lock()  # one reconstruction at a time on the GPU

    def __call__(self, image_paths: list[str]) -> Reconstruction:
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch = self.torch
        with self.lock:
            images = load_and_preprocess_images(image_paths).to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
                pred = self.model(images)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
            extrinsic = extrinsic[0].float().cpu().numpy()
            intrinsic = intrinsic[0].float().cpu().numpy()
            depth = pred["depth"][0].float().cpu().numpy()
            conf = pred["depth_conf"][0].float().cpu().numpy()
            colors = images.permute(0, 2, 3, 1).float().cpu().numpy()
            del pred, images
            torch.cuda.empty_cache()
        return Reconstruction(extrinsic, unproject(depth, extrinsic, intrinsic), conf, colors)


class OmegaRunner:
    """VGGT-Omega from a .pt checkpoint; it does its own mixed precision."""

    def __init__(self, path: str, resolution: int = 512, device: str = "cuda"):
        import torch
        from vggt_omega.models import VGGTOmega

        self.torch = torch
        self.device = device
        self.resolution = resolution
        self.model = VGGTOmega().to(device).eval()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.lock = threading.Lock()

    def __call__(self, image_paths: list[str]) -> Reconstruction:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        torch = self.torch
        with self.lock:
            images = load_and_preprocess_images(image_paths, image_resolution=self.resolution).to(self.device)
            with torch.inference_mode():
                pred = self.model(images)
            extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], images.shape[-2:])
            extrinsic = extrinsic[0].float().cpu().numpy()
            intrinsic = intrinsic[0].float().cpu().numpy()
            depth = pred["depth"][0].float().cpu().numpy()
            conf = pred["depth_conf"][0].float().cpu().numpy()
            colors = images.permute(0, 2, 3, 1).float().cpu().numpy()
            del pred, images
            torch.cuda.empty_cache()
        return Reconstruction(extrinsic, unproject(depth, extrinsic, intrinsic), conf, colors)


def load(path: str, resolution: int = 512):
    """A VGGT-Omega .pt checkpoint, or a VGGT-1B folder."""
    return OmegaRunner(path, resolution) if Path(path).suffix == ".pt" else VGGTRunner(path)
