import argparse
import os
import sys

# Import this in scripts to access self-made packages (ie. src.utils)
current_dir = os.getcwd()
sys.path.append(current_dir)

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from src.utils import (
    load_config,
    load_module_from_config,
    get_log_dir,
    load_datamodule_from_config,
)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--fold-indices", nargs="+", type=int, default=None)
    args = parser.parse_args()
    config_filepath = str(args.config)
    cfg = load_config(filepath=config_filepath)

    log_dir_base = get_log_dir(cfg, config_filepath)

    num_folds = cfg.cv_folds if "cv_folds" in cfg else 1
    fold_indices = (
        list(range(num_folds))
        if args.fold_indices is None
        else [int(fold_idx) for fold_idx in args.fold_indices]
    )
    invalid_folds = [
        fold_idx
        for fold_idx in fold_indices
        if fold_idx < 0 or fold_idx >= num_folds
    ]
    if invalid_folds:
        raise ValueError(
            f"Fold indices outside [0, {num_folds}): {invalid_folds}"
        )

    for fold_idx in fold_indices:
      print(f"=== Fold {fold_idx + 1}/{num_folds} ===")

      fold_log_dir = f"{log_dir_base}/fold_{fold_idx}"

      if num_folds == 1:
        fold_log_dir = log_dir_base

      seed_everything(314159 + fold_idx, workers=True)
      data_module = load_datamodule_from_config(cfg, fold_idx=fold_idx)
      model = load_module_from_config(cfg)

      # define callback for selecting checkpoints during training
      checkpoint_callback = ModelCheckpoint(
          dirpath=fold_log_dir + "/checkpoints",
          filename="{epoch}_{val_loss}",
          verbose=True,
          monitor="val_loss",
          mode="min",
      )

      trainer_kwargs = {}
      if cfg.if_cuda and torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = cfg.num_gpus if "num_gpus" in cfg and cfg.num_gpus > 0 else 1

      # define trainer
      trainer = Trainer(
          max_epochs=cfg.epochs,
          deterministic=True,
          default_root_dir=fold_log_dir,
          val_check_interval=1.0,
          callbacks=[checkpoint_callback],
          **trainer_kwargs,
      )

      trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
