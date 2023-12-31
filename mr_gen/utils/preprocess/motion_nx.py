from typing import Dict
import torch
import numpy as np


class MotionPreprocessorNX:
    def __init__(self, cfg):
        self.cfg = cfg
        self.delta_order: int = cfg.delta_order
        self.use_centroid: bool = cfg.use_centroid
        self.use_angle: bool = cfg.use_angle
        self.train_by_std: bool = cfg.train_by_std

    def __call__(
        self,
        npz_path: str,
        start: int,
        end: int,
        stride: int,
    ) -> torch.Tensor:
        start += stride - 1
        end += stride - 1

        head_seq = []
        angle_centroid: Dict[str, np.ndarray] = np.load(npz_path)
        angle = angle_centroid["angle"][start:end:stride]
        centroid = angle_centroid["centroid"][start:end:stride]
        if not self.train_by_std:
            angle *= angle_centroid["angle_std"]
            angle += angle_centroid["angle_mean"]

            centroid *= angle_centroid["centroid_std"]
            centroid += angle_centroid["centroid_mean"]

        angle = torch.tensor(angle)
        centroid = torch.tensor(centroid)

        head_seq = torch.cat([angle, centroid], dim=-1).to(torch.float32)
        head_seq_with_delta = self.compute_delta(head_seq)

        msg = f"start: {start}, end: {end}, stride: {stride}, len: {len(angle_centroid['angle'])}"
        assert len(head_seq_with_delta) != 0, msg + "\n" + npz_path

        return head_seq_with_delta

    def compute_delta(self, head_seq: torch.Tensor) -> torch.Tensor:
        if self.delta_order == 0:
            return head_seq

        delta1 = head_seq[1:] - head_seq[:-1]
        if self.delta_order == 1:
            return torch.cat([head_seq[1:], delta1], dim=1)

        delta2 = delta1[1:] - delta1[:-1]
        if self.delta_order == 2:
            return torch.cat([head_seq[2:], delta1[1:], delta2], dim=1)

        raise ValueError("delta_order must be 0, 1 or 2")
