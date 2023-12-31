import os
import json
from typing import Any, Dict, List
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import Dataset

from mr_gen.utils.visualize.dataloader.audio import AudioPreprocessor
from mr_gen.utils.visualize.dataloader.motion_nx import MotionPreprocessorNX


# -100 is not appeared in the dataset (both of standalized and non-standalized)
PADDING_VALUE = -100


class HeadMotionDatasetNX(Dataset):
    def __init__(
        self, dataset_path: str, motion: DictConfig, audio: DictConfig
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.data_list = self.load_segment_list()
        self.motion = motion
        self.audio = audio

        self.audio_preprocessor = AudioPreprocessor(audio)
        self.motion_preprocessor = MotionPreprocessorNX(motion)

    def __getitem__(self, index: int):
        jdic_path = self.data_list[index]
        with open(jdic_path, "r", encoding="utf-8") as f:
            for lines, line in enumerate(f):
                if lines > 0:
                    raise ValueError("json file must have only one line.")
                jdic = json.loads(line)

        partner_motion = jdic["partner_motion"]
        partner_audio = jdic["partner_audio"]
        self_motion = jdic["self_motion"]
        target = jdic["target"]

        offset_p = partner_motion["offset"]
        offset_s = self_motion["offset"]

        # load main data
        fbank_partner = self.audio_preprocessor(
            partner_audio["path"],
            partner_audio["seq"]["start"],
            partner_audio["seq"]["end"],
        )
        motion_partner = self.motion_preprocessor(
            partner_motion["path"],
            partner_motion["seq"]["start"] - offset_p,
            partner_motion["seq"]["end"] - offset_p,
            partner_motion["seq"]["stride"],
            jdic_path,
        )
        motion_self = self.motion_preprocessor(
            self_motion["path"],
            self_motion["seq"]["start"] - offset_s,
            self_motion["seq"]["end"] - offset_s,
            self_motion["seq"]["stride"],
            jdic_path,
        )

        # load leading data
        leading_fbank_partner = self.audio_preprocessor(
            partner_audio["path"],
            partner_audio["lead"]["start"],
            partner_audio["lead"]["end"],
        )
        leading_motion_partner = self.motion_preprocessor(
            partner_motion["path"],
            partner_motion["lead"]["start"] - offset_p,
            partner_motion["lead"]["end"] - offset_p,
            partner_motion["lead"]["stride"],
            jdic_path,
        )
        leading_motion_self = self.motion_preprocessor(
            self_motion["path"],
            self_motion["lead"]["start"] - offset_s,
            self_motion["lead"]["end"] - offset_s,
            self_motion["lead"]["stride"],
            jdic_path,
        )

        target_shift = target["shift_input_seq"]
        target_r_shift = target["shift_real_seq"]

        (t_tensor, l), t_st, t_path, sss = motion_self
        t_sss = (sss[0] + target_r_shift, sss[1] + target_r_shift, sss[2])
        target = (t_tensor[:, target_shift:], l), t_st, t_path, t_sss

        ms_len = t_tensor.shape[1] - target_shift
        motion_self = (t_tensor[:, :ms_len], l), t_st, t_path, sss

        return (
            fbank_partner,
            motion_partner,
            motion_self,
            leading_fbank_partner,
            leading_motion_partner,
            leading_motion_self,
            target,
        )

    def __len__(self) -> int:
        return len(self.data_list)

    def load_segment_list(self) -> List[Dict[str, Any]]:
        dict_datasets = []
        for path in tqdm(
            os.listdir(self.dataset_path), desc="load segment list", position=0
        ):
            if path.endswith(".json"):
                dict_datasets.append(os.path.join(self.dataset_path, path))
        return dict_datasets


def collate_fn(batch: List[tuple]):
    batch = list(batch[0])

    return batch
