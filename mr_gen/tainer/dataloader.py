import os
from logging import Logger
import json
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from mr_gen.databuild import DataBuilderNX
from mr_gen.utils.preprocess import AudioPreprocessor, MotionPreprocessorNX


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

        # load main data
        fbank_partner = self.audio_preprocessor(
            partner_audio["path"],
            partner_audio["seq"]["start"],
            partner_audio["seq"]["end"],
        )
        motion_partner = self.motion_preprocessor(
            partner_motion["path"],
            partner_motion["seq"]["start"],
            partner_motion["seq"]["end"],
            partner_motion["seq"]["stride"],
        )
        motion_self = self.motion_preprocessor(
            self_motion["path"],
            self_motion["seq"]["start"],
            self_motion["seq"]["end"],
            self_motion["seq"]["stride"],
        )

        # load leading data
        leading_fbank_partner = self.audio_preprocessor(
            partner_audio["path"],
            partner_audio["lead"]["start"],
            partner_audio["lead"]["end"],
        )
        leading_motion_partner = self.motion_preprocessor(
            partner_motion["path"],
            partner_motion["lead"]["start"],
            partner_motion["lead"]["end"],
            partner_motion["lead"]["stride"],
        )
        leading_motion_self = self.motion_preprocessor(
            self_motion["path"],
            self_motion["lead"]["start"],
            self_motion["lead"]["end"],
            self_motion["lead"]["stride"],
        )

        target = motion_self[target["shift"] :]
        motion_self = motion_self[: len(motion_self) - target["shift"]]

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


def collate_fn(batch):
    # fbank, motion_context, motion_target = zip(*batch)
    modals = zip(*batch)
    modals = [torch.stack(modal, dim=0) for modal in modals]
    return modals


class HeadMotionDataModuleNX(pl.LightningDataModule):
    def __init__(
        self,
        data: DictConfig,
        exp: DictConfig,
        audio: DictConfig,
        motion: DictConfig,
        logger: Logger,
    ) -> None:
        super().__init__()
        self.data_dir = data.data_dir
        self.batch_size = exp.batch_size
        self.train_rate = exp.train_rate
        self.valid_rate = exp.valid_rate

        self.data = data
        self.exp = exp
        self.audio = audio
        self.motion = motion
        self.logger = logger

        self.dataset_path: str
        self.train_size: int
        self.valid_size: int
        self.test_size: int

        self.dataset: HeadMotionDatasetNX
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_path = DataBuilderNX(self.data, self.logger).data_site

        self.dataset = HeadMotionDatasetNX(self.dataset_path, self.motion, self.audio)
        self.train_size = int(self.train_rate * len(self.dataset))
        self.valid_size = int(self.valid_rate * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.valid_size

        if self.test_size <= 0:
            raise ValueError("test size is negative or zero.")

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [self.train_size, self.valid_size + self.test_size]
        )
        self.val_dataset, self.test_dataset = random_split(
            self.val_dataset, [self.valid_size, self.test_size]
        )

        self.logger.info(
            f"train size: {self.train_size}, valid size: {self.valid_size}, test size: {self.test_size}"
        )
        self.logger.info(f"dataset path: {self.dataset_path}")
        self.logger.info("Dataset ready.")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            shuffle=True,
        )
