import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torchviz import make_dot

from mr_gen.utils.logger_gen import set_logger
from mr_gen.model.lstmformer import Metaformer
from mr_gen.model.lstmformer.dataloader import HeadMotionDataModuleNX


def setup_logger(args: DictConfig):
    if args.exp.use_logger != "wandb":
        return None

    logger = WandbLogger(
        name=args.logger.exp_name,
        project=args.logger.project,
        save_dir=args.logger.save_dir,
        version=args.logger.version,
    )
    return logger


def setup_callbacks(args: DictConfig):
    callbacks = []
    if args.callbacks.use_checkpoint:
        model_ckpt = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=args.callbacks.save_top_k,
            save_last=True,
            filename="V{epoch:03d}-{val_loss:.2f}",
        )
        callbacks.append(model_ckpt)
        model_ckpt_gen = ModelCheckpoint(
            monitor="genrt_loss",
            mode="min",
            save_top_k=args.callbacks.save_top_k,
            save_last=False,
            filename="G{epoch:03d}-{genrt_loss:.2f}",
        )
        callbacks.append(model_ckpt_gen)
        model_ckpt_train = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=args.callbacks.save_top_k,
            save_last=False,
            filename="T{epoch:03d}-{train_loss:.2f}",
        )
        callbacks.append(model_ckpt_train)
    if args.callbacks.use_early_stopping:
        patience = args.callbacks.patience_epoch / args.trainer.val_check_interval
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        )
        callbacks.append(early_stop)
    if args.exp.use_logger == "wandb":
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    if callbacks == []:
        callbacks = None
    return callbacks


def visualize_model(cfg):
    device = "cuda:0" if torch.cuda.is_available() and cfg.device == "gpu" else "cpu"
    model = Metaformer(cfg.model, cfg.optim, cfg.metrics).to(device)

    if cfg.model.visualize:
        motion = torch.randn(1, 1, 18).to(device)
        audio = torch.randn(1, 1, 81).to(device)
        inputs = (
            (audio, 1),
            (motion, 1),
            (motion.clone(), 1),
            (audio.clone(), 1),
            (motion.clone(), 1),
            (motion.clone(), 1),
            None,
        )
        y, _ = model.forward(*inputs)
        img_name = cfg.model.visualize_name
        img = make_dot(y, params=dict(model.named_parameters()))
        img.format = "png"
        img.render(img_name)

    del model


@hydra.main(version_base=None, config_path="run/lstmformer/")  # type: ignore
def main(cfg: DictConfig):
    # set logger and callbacks
    logger = setup_logger(cfg)
    callbacks = setup_callbacks(cfg)

    lounch_logger = set_logger(
        name="launch",
        use_handler=False,
    )

    device = "cuda:0" if torch.cuda.is_available() and cfg.device == "gpu" else "cpu"

    model = Metaformer(cfg.model, cfg.optim, cfg.metrics).to(device)
    datamodule = HeadMotionDataModuleNX(
        cfg.data, cfg.exp, cfg.audio, cfg.motion, lounch_logger
    )

    visualize_model(cfg)

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    lounch_logger.info("Finish setup.")
    lounch_logger.info("Start training.")

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()  # type: ignore # pylint: disable=no-value-for-parameter
