from logging import Logger
import json
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from mr_gen.databuild import DataBuilder
from mr_gen.utils.preprocess import AudioPreprocessor, MotionPreprocessor


class HeadMotionDataset(Dataset):
    def __init__(self, dataset_path: str, cfg: DictConfig, audio: DictConfig) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.data_list = self.load_segment_list()
        self.cfg = cfg
        self.audio = audio

        self.audio_preprocessor = AudioPreprocessor(audio)
        self.motion_preprocessor = MotionPreprocessor(cfg)

    def __getitem__(self, index: int):
        # return self.dataset_path[index]
        jdic = self.data_list[index]

        fbank = self.audio_preprocessor(
            jdic["audio_path"], jdic["audio"]["start"], jdic["audio"]["end"]
        )
        motion_context = self.motion_preprocessor(jdic["head_dir"], **jdic["context"])
        motion_target = self.motion_preprocessor(jdic["head_dir"], **jdic["target"])

        return fbank, motion_context, motion_target

    def __len__(self) -> int:
        return len(self.dataset_path)

    def load_segment_list(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dict_dataset = []
            for line in f:
                dict_dataset.append(json.loads(line))
            return dict_dataset


class HeadMotionDataModule(pl.LightningDataModule):
    def __init__(
        self, cfg: DictConfig, exp: DictConfig, audio: DictConfig, logger: Logger
    ) -> None:
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = exp.batch_size
        self.train_rate = exp.train_rate
        self.valid_rate = exp.valid_rate

        self.cfg = cfg
        self.exp = exp
        self.audio = audio
        self.logger = logger

        self.data_build: DataBuilder
        self.dataset_path: str
        self.train_size: int
        self.valid_size: int
        self.test_size: int

        self.dataset: HeadMotionDataset
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.data_build = DataBuilder(self.cfg, self.logger)
        self.dataset_path = self.data_build.data_file

        self.dataset = HeadMotionDataset(self.dataset_path, self.cfg, self.audio)
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
