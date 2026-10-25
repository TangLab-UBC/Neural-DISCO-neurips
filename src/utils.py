import glob
import os
from pathlib import Path
import shutil

from src.modules.neural_disco_module import NeuralDISCOModule
from src.modules.gauss_glm_module import GaussianGLMModule
import yaml
import torch
from munch import munchify
from src.modules.DataModule import (
    MCMazeDataModule,
    NLB_DATASETS,
    AllenDataModule,
    CustomRatInABoxDataModule,
)

def load_config(filepath):
    with open(filepath, "r") as stream:
        cfg = yaml.safe_load(stream)
    return munchify(cfg)


def mkdir(folder, overwrite=False):
    if overwrite:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder, exist_ok=True)


def get_config_subpath(config_filepath):
    path = Path(config_filepath)
    # Join the directories below "configs" and the file stem: configs/a/b/c.yaml -> a-b-c
    parts = path.parts
    start = len(parts) - parts[::-1].index("configs") if "configs" in parts else 1
    subpath = "-".join(parts[start:-1]) + '-' + path.stem
    return subpath


def get_log_dir(cfg, config_filepath):
    log_dir = "logs/" + get_config_subpath(config_filepath)
    return log_dir


def load_datamodule_from_config(cfg, fold_idx=0):
    common_kwargs = dict(
        if_cuda=cfg.if_cuda,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        num_folds=cfg.cv_folds if "cv_folds" in cfg else 10,
        fold_idx=fold_idx,
        num_workers=cfg.num_workers,
        input_window=cfg.input_window,
        output_window=cfg.output_window,
    )

    if cfg.dataset == 'allen_vc':
      return AllenDataModule(
        **common_kwargs,
        normalize_targets=cfg.normalize_targets,
        bin_width=cfg.bin_width if "bin_width" in cfg else 50,
        session_id=cfg.session_id,
        num_synth_neurons=cfg.num_synth_neurons if 'num_synth_neurons' in cfg else 0,
        held_out_visual_stimuli=cfg.held_out_visual_stimuli if 'held_out_visual_stimuli' in cfg else {},
    )

    if cfg.dataset in NLB_DATASETS:
        return MCMazeDataModule(
            **common_kwargs,
            smoothing_constant=(cfg.smoothing_constant if "smoothing_constant" in cfg else 5.0),
            normalize_targets=cfg.normalize_targets,
            normalize_spikes=cfg.normalize_spikes,
            bin_width=cfg.bin_width if "bin_width" in cfg else 1,
            behaviours=cfg.behaviours if "behaviours" in cfg else None,
            target_normalization_mode=(
                cfg.target_normalization_mode if "target_normalization_mode" in cfg else None
            ),
            delay_factor=(cfg.delay_factor if "delay_factor" in cfg else 80),
            unknown_input_window=(
                cfg.unknown_input_window if "unknown_input_window" in cfg else None
            ),
            spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
            behaviour_target_mode=(
                cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None
            ),
            num_synth_neurons=(cfg.num_synth_neurons if "num_synth_neurons" in cfg else 0),
        )

    if cfg.dataset == "custom_rat_in_a_box.pkl":
        return CustomRatInABoxDataModule(
            **common_kwargs,
            smoothing_constant=(cfg.smoothing_constant if "smoothing_constant" in cfg else 5.0),
            normalize_targets=cfg.normalize_targets,
            normalize_spikes=cfg.normalize_spikes,
            bin_width=cfg.bin_width if "bin_width" in cfg else 1,
            num_cells=cfg.num_cells if "num_cells" in cfg else 100,
            noise=cfg.noise if "noise" in cfg else 0.1,
            minutes=cfg.minutes if "minutes" in cfg else 5,
            num_bins=cfg.num_bins if "num_bins" in cfg else 11,
            use_counterfactual=(cfg.use_counterfactual if "use_counterfactual" in cfg else False),
            manual_trajectory=(cfg.manual_trajectory if "manual_trajectory" in cfg else False),
            include_noise_cell=(cfg.include_noise_cell if "include_noise_cell" in cfg else False),
            delay_factor=(cfg.delay_factor if "delay_factor" in cfg else 80),
            unknown_input_window=(
                cfg.unknown_input_window if "unknown_input_window" in cfg else None
            ),
            spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
            behaviour_target_mode=(
                cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None
            ),
        )

    raise ValueError(f"Unsupported dataset for datamodule loading: {cfg.dataset}")



def load_module_from_config(cfg):
    num_neurons = cfg.num_neurons
    if "num_synth_neurons" in cfg:
        num_neurons += cfg.num_synth_neurons

    rat_behaviours = None
    rat_behaviour_dims = None
    if getattr(cfg, "dataset", None) == "custom_rat_in_a_box.pkl":
        rat_behaviours = ["pos", "head_direction"]
        rat_behaviour_dims = {"pos": 2, "head_direction": 2}

    if getattr(cfg, "gaussian-glm", False) is True:
        return GaussianGLMModule(
            lr=cfg.lr,
            if_cuda=cfg.if_cuda,
            gamma=cfg.gamma,
            weight_decay=cfg.weight_decay if "weight_decay" in cfg else 0.0,
            num_workers=cfg.num_workers,
            lr_schedule=cfg.lr_schedule if "lr_schedule" in cfg else None,
            input_window=cfg.input_window if "input_window" in cfg else 1,
            output_window=cfg.output_window if "output_window" in cfg else 1,
            dataset=cfg.dataset,
            num_neurons=num_neurons,
            reconstruction_target_key=(
                cfg.reconstruction_target_key
                if "reconstruction_target_key" in cfg
                else "spikes"
            ),
            behaviours=cfg.behaviours if "behaviours" in cfg else rat_behaviours,
            behaviour_dims=(
                cfg.behaviour_dims
                if "behaviour_dims" in cfg
                else rat_behaviour_dims
            ),
        )

    model = NeuralDISCOModule(
        lr=cfg.lr,
        if_cuda=cfg.if_cuda,
        gamma=cfg.gamma,
        num_workers=cfg.num_workers,
        lr_schedule=cfg.lr_schedule,
        latent_dim=cfg.latent_dim,
        weight_decay=cfg.weight_decay,
        l2_unknown=(
            cfg.l2_unknown if "l2_unknown" in cfg else 0.0
        ),
        l1_alpha=(
            cfg.l1_alpha if "l1_alpha" in cfg else 0.0
        ),
        num_layers=cfg.num_layers,
        input_window=cfg.input_window,
        output_window=cfg.output_window,
        num_neurons=num_neurons,
        dataset=cfg.dataset,
        reconstruction_style=cfg.reconstruction_style if "reconstruction_style" in cfg else "MSE",
        reconstruction_target_key=(
          cfg.reconstruction_target_key if 'reconstruction_target_key' in cfg else 'spikes'
        ),
        unknown_attribute_latent_dim=(
            cfg.unknown_attribute_latent_dim if "unknown_attribute_latent_dim" in cfg else 32
        ),
        unknown_sigma=(
          cfg.unknown_sigma if "unknown_sigma" in cfg else 0.0
        ),
        mlp=(
          cfg.mlp if 'mlp' in cfg else True
        ),
        encoder_loss_alpha=(
          cfg.encoder_loss_alpha if 'encoder_loss_alpha' in cfg else 1.0
        ),
        reconstruction_penalty=(
          cfg.reconstruction_penalty if 'reconstruction_penalty' in cfg else 0.0
        ),
        l2_warmup_epochs=(
          cfg.l2_warmup_epochs if 'l2_warmup_epochs' in cfg else 0
        ),
        stage0_unknown_off_epochs=(
          cfg.stage0_unknown_off_epochs if 'stage0_unknown_off_epochs' in cfg else 0
        ),
        always_zero_unknown=(
          cfg.always_zero_unknown if 'always_zero_unknown' in cfg else False
        ),
        unknown_input_key=(
          cfg.unknown_input_key if 'unknown_input_key' in cfg else 'spikes'
        ),
        unknown_input_window=(
          cfg.unknown_input_window if 'unknown_input_window' in cfg else None
        ),
        unknown_model_type=(
          cfg.unknown_model_type if 'unknown_model_type' in cfg else None
        ),
        behaviour_num_layers=(
          cfg.behaviour_num_layers if 'behaviour_num_layers' in cfg else None
        ),
        unknown_num_layers=(
          cfg.unknown_num_layers if 'unknown_num_layers' in cfg else None
        ),
        decoder_num_layers=(
          cfg.decoder_num_layers if 'decoder_num_layers' in cfg else None
        ),
        behaviours=cfg.behaviours if "behaviours" in cfg else rat_behaviours,
        behaviour_dims=cfg.behaviour_dims if "behaviour_dims" in cfg else rat_behaviour_dims,
        behaviour_model_types=cfg.behaviour_model_types if "behaviour_model_types" in cfg else None,
    )
    return model


def load_model_from_ckpt(cfg, log_dir):
    # Loads the newest checkpoint; train.py keeps only the one with the lowest val_loss.
    filepath = log_dir + "/checkpoints/"
    ckpt_files = glob.glob(os.path.join(filepath, "*.ckpt"))
    checkpoint_filepath = max(ckpt_files, key=os.path.getmtime)
    module = load_module_from_config(cfg)
    import munch
    torch.serialization.add_safe_globals([munch.Munch])
    ckpt = torch.load(checkpoint_filepath, map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"])
    module.eval()
    module.requires_grad_(False)
    return module.model


def load_cv_models(cfg, base_dir):
    return list(map(lambda subdir: load_model_from_ckpt(cfg, subdir), get_cv_subdirs(cfg, base_dir)))

def get_cv_subdirs(cfg, base_dir):
    num_folds = cfg.cv_folds
    subdirs = []
    for fold_idx in range(num_folds):
      subdirs.append(f"{base_dir}/fold_{fold_idx}")
    return subdirs


def get_visualization_filepath(log_dir, file_name):
    vis_dir = log_dir + "/visualizations"
    mkdir(vis_dir)
    return vis_dir + "/" + file_name


def get_result_filepath(log_dir, file_name):
    res_dir = log_dir + "/results"
    mkdir(res_dir)
    return res_dir + "/" + file_name

def get_custom_filepath(log_dir, file_name, folder_name_1, folder_name_2=None, folder_name_3=None, folder_name_4=None, folder_name_5=None, folder_name_6=None):
    dir = os.path.join(log_dir, folder_name_1)
    mkdir(dir)
    if folder_name_2 is not None:
        dir = os.path.join(dir, folder_name_2)
        mkdir(dir)
        if folder_name_3 is not None:
            dir = os.path.join(dir, folder_name_3)
            mkdir(dir)
            if folder_name_4 is not None:
                dir = os.path.join(dir, folder_name_4)
                mkdir(dir)
                if folder_name_5 is not None:
                    dir = os.path.join(dir, folder_name_5)
                    mkdir(dir)
                    if folder_name_6 is not None:
                        dir = os.path.join(dir, folder_name_6)
                        mkdir(dir)
    dir = os.path.join(dir, file_name)
    return dir
