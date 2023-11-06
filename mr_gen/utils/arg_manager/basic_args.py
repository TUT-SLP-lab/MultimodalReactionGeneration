from argparse import ArgumentParser
from typing import Union


DEFO_PROJ_NAME = "Multimodal-Head-Motion-Prediction"


class TrainerArgmentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("--accelerator", type=str, default="auto")
        self.add_argument("--accumulate-grad-batches", type=int, default=1)
        self.add_argument("--benchmark", action="store_true")
        self.add_argument("--deterministic", action="store_true")
        self.add_argument("--check-val-every-n-epoch", type=int, default=1)
        self.add_argument("--default-root-dir", type=str, default="./logs")
        self.add_argument("--devices", type=Union[int, str], nargs="+", default="auto")
        self.add_argument("--enable-checkpointing", action="store_true")
        self.add_argument("--max-epochs", type=int, default=100)
        self.add_argument("--gpus", type=int, default=1)
        self.add_argument("--log-every-n-steps", type=int, default=50)
        self.add_argument("--precision", type=int, default=16)
        self.add_argument("--strategy", type=str, default="ddp")
        self.add_argument("--val-check-interval", type=float, default=0.25)

        self.add_argument("--ckpt-path", type=str, default="./ckpts")

        self.add_argument("--exp-name", type=str, default="cradle-01")
        self.add_argument("--project", type=str, default=DEFO_PROJ_NAME)
        self.add_argument("--save-dir", type=str, default="./log")
        self.add_argument("--version", type=str, default="0.0.1")

        self.add_argument("--save-top-k", type=int, default=5)
        self.add_argument("--patience-epoch", type=int, default=10)


class TrainArgmentParser(TrainerArgmentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("--use-model", type=str, default="simple_lstm")
        self.add_argument("--use-optimizer", type=str, default="adam")
        self.add_argument("--use-logger", type=str, default="wandb")
        self.add_argument("--use-checkpoint", action="store_true")
        self.add_argument("--use-early-stopping", action="store_true")
        self.add_argument("--use-amp", action="store_true")
        self.add_argument("--train-rate", type=float, default=0.9)
        self.add_argument("--valid-rate", type=float, default=0.05)

        self.add_argument("--momentum", type=float, default=0.9)
        self.add_argument("--weight-decay", type=float, default=0)
        self.add_argument("--lr", type=float, default=1e-6)
        self.add_argument("--use-lr-sched", action="store_true")
        self.add_argument("--batch-size", type=int, default=64)


class DataArgmentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("--data-dir", type=str, default="./data/dataset")
        self.add_argument("--fps", type=float, default=25)
        self.add_argument("--context-start", type=int, default=-30)
        self.add_argument("--sample-stride", type=int, default=2)
        self.add_argument("--context-size", type=int, default=15)
        self.add_argument("--context-stride", type=int, default=2)
        self.add_argument("--target-type", type=str, default="direct")
        self.add_argument("--target-position", type=int, default=0)
        self.add_argument("--target-size", type=int, default=1)
        self.add_argument("--target-stride", type=int, default=1)
        self.add_argument("--delta-order", type=int, default=2)
        self.add_argument("--use-centroid", action="store_true")
        self.add_argument("--use-angle", action="store_true")
        self.add_argument("--no-cache-build", action="store_true")
        self.add_argument("--clear-cache", action="store_true")


class PreprocessArgmentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("--sample-rate", type=int, default=16000)
        self.add_argument("--nfft", type=int, default=400)
        self.add_argument("--shift", type=int, default=160)
        self.add_argument("--nmels", type=int, default=25)


class IntegratedBasicArgmentParser(
    DataArgmentParser, TrainArgmentParser, PreprocessArgmentParser
):
    ...
