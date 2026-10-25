import os
import sys
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import default_collate
# Import this in scripts to access self-made packages (ie. src.utils)
current_dir = os.getcwd()
sys.path.append(current_dir)

from src.utils import (
    get_result_filepath,
    load_model_from_ckpt,
    get_log_dir,
    load_config,
    load_datamodule_from_config,
    load_cv_models,
    get_cv_subdirs
)

def attribute(log_dir, model, datamodule, fold_idx=0):
    # 32 fold-training samples, the same for every lambda in the sweep
    rows = np.random.default_rng(271828 + fold_idx).permutation(datamodule.train_dataset.indices)[:32]
    batch = default_collate([datamodule.train_dataset.dataset[int(row)] for row in rows])
    attributions = model.attribute(batch)

    save_path = get_result_filepath(log_dir, 'attribution_scores.pt')

    torch.save(attributions, save_path)



if __name__ == "__main__":
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    log_dir = get_log_dir(cfg, config_filepath)
    if 'cv_folds' in cfg and cfg.cv_folds > 1:
      cv_subdirs = get_cv_subdirs(cfg, log_dir)
      cv_models = load_cv_models(cfg, log_dir)
      for fold_idx, (fold_model, fold_dir) in enumerate(zip(cv_models, cv_subdirs)):
        # Same seed as training, so MC Maze control neurons match the trained model's
        seed_everything(314159 + fold_idx)
        datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx)
        attribute(fold_dir, fold_model, datamodule, fold_idx)
    else:
      seed_everything(314159)
      datamodule = load_datamodule_from_config(cfg)
      model = load_model_from_ckpt(cfg, log_dir)
      attribute(log_dir, model, datamodule)
