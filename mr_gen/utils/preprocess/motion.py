import os
import pickle
from typing import Any

import torch

from mr_gen.utils.io import ZERO_PADDING
from mr_gen.utils.tools.adapter import FaceAdapter


class MotionPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.delta_order = cfg.delta_order
        self.use_centroid = cfg.use_centroid
        self.use_angle = cfg.use_angle

    def __call__(
        self,
        head_dir: str,
        start: int,
        end: int,
        stride: int,
    ) -> Any:
        head_seq = []
        for idx in range(start, end, stride):
            base_name = os.path.split(head_dir)[1]
            idx_str = str(idx).zfill(ZERO_PADDING)
            file_name = base_name + "_" + idx_str + ".head"
            target_path = os.path.join(head_dir, file_name)

            with open(target_path, "rb") as f:
                head: FaceAdapter = pickle.load(f)[1]  # (idx, head)

            record = []
            if self.use_centroid:
                centroid = (head.centroid - head.centroid_mean) / head.centroid_std
                record.append(torch.tensor(centroid))
            if self.use_angle:
                # followed related work "BLSTM Neural Networks for Speech-Driven Head Motion Synthesis"
                # Ding et al. 2015
                angle = (head.angle - head.angle_mean) / head.angle_std
                record.append(torch.tensor(angle))
            if len(record) == 0:
                raise ValueError(
                    "Specify at least one of --use-centroid or --use-angle"
                )
            record = torch.cat(record, dim=0)
            head_seq.append(record)

        head_seq = torch.stack(head_seq, dim=0).to(torch.float32)
        head_seq_with_delta = self.compute_delta(head_seq)

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
