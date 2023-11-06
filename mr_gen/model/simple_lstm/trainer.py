import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mr_gen.utils.logger_gen import set_logger

from mr_gen.model.simple_lstm import SimpleLSTM
from mr_gen.model.simple_lstm.dataloader import HeadMotionDataModule


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
            filename="{epoch}-{val_loss:.2f}",
        )
        callbacks.append(model_ckpt)
    if args.callbacks.use_early_stopping:
        patience = args.callbacks.patience_epoch / args.trainer.val_check_interval
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        )
        callbacks.append(early_stop)
    if callbacks == []:
        callbacks = None
    return callbacks


@hydra.main(version_base=None, config_path="run/simple_lstm/")  # type: ignore
def main(cfg: DictConfig):
    # set logger and callbacks
    logger = setup_logger(cfg)
    callbacks = setup_callbacks(cfg)

    lounch_logger = set_logger(
        name="lounch", rootname=cfg.logger.save_dir + "/lounch.log"
    )

    model = SimpleLSTM(cfg.model, cfg.optim, cfg.metrics)
    datamodule = HeadMotionDataModule(cfg.data, cfg.exp, cfg.audio, lounch_logger)

    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)

    lounch_logger.info("Finish setup.")
    lounch_logger.info("Start training.")

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()  # type: ignore # pylint: disable=no-value-for-parameter
