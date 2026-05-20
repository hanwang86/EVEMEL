import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf

from model.lightning_eve import LightningForEVEMEL
from utils.dataset import RawMentionDataModule
from utils.dataset_ama import ToolAdaptiveMentionDataModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/richpediamel_ama.yaml', help='Path to the config file')
    cmd_args = parser.parse_args()

    print(f"Loading config from: {cmd_args.config}")
    if not os.path.exists(cmd_args.config):
        raise FileNotFoundError(f"Config file not found at: {cmd_args.config}")

    args = OmegaConf.load(cmd_args.config)

    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    if 'grounded_img_folder' in args.data:
        data_module = ToolAdaptiveMentionDataModule(args)
    else:
        data_module = RawMentionDataModule(args)

    lightning_model = LightningForEVEMEL(args)

    logger = pl.loggers.TensorBoardLogger("tb_logs", name=args.run_name)

    ckpt_callbacks = ModelCheckpoint(monitor='Val/mrr', save_weights_only=True, mode='max')
    early_stop_callback = EarlyStopping(monitor="Val/mrr", min_delta=0.00, patience=3, verbose=True, mode="max")
    callbacks = [ckpt_callbacks, early_stop_callback]

    trainer = pl.Trainer(
        **args.trainer,
        deterministic=True,
        logger=logger,
        default_root_dir="./runs",
        callbacks=callbacks,
    )

    if args.data.test_ckpt is not None:
        print(f"\n[Test Mode] Loading checkpoint for evaluation: {args.data.test_ckpt}")
        trainer.test(
            model=lightning_model,
            datamodule=data_module,
            ckpt_path=args.data.test_ckpt,
        )
    else:
        print("\n[Train Mode] Training EVEMEL from scratch...")
        trainer.fit(model=lightning_model, datamodule=data_module)

        print("\nTraining complete. Evaluating with the best checkpoint...")
        trainer.test(
            model=lightning_model,
            datamodule=data_module,
            ckpt_path='best',
        )
