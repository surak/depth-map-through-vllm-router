"""Run VGGT on a list of image files."""

from __future__ import annotations

import threading

import numpy as np

from .mapping import Reconstruction


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
        from vggt.utils.geometry import unproject_depth_map_to_point_map
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
        points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        return Reconstruction(extrinsic, np.asarray(points, np.float32), conf, colors)
