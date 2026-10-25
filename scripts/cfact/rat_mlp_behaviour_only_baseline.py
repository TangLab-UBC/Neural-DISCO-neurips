import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.plotting import COUNTERFACTUAL_COLORS, COUNTERFACTUAL_SUMMARY_COLORS  # noqa: E402
from src.data.collate import collate_all_batches  # noqa: E402
from src.utils import (  # noqa: E402
    get_log_dir,
    get_result_filepath,
    get_visualization_filepath,
    load_config,
    load_datamodule_from_config,
    load_model_from_ckpt,
)


DEFAULT_MAIN_CONFIG = "configs/rat_in_a_box_custom/mlp_counterfactual_spikes_iw1_cv.yaml"

COUNTERFACTUAL_MODES = {
    "position_shift": {
        "source_flag": "shift_left",
        "target_flag": "shift_right",
        "target_reverse": False,
        "title": "Position-shift reconstruction",
        "suptitle": "RatInABox position shift with held-out speed",
        "stem_prefix": "",
    },
    "inverse_trajectory": {
        "source_flag": "val",
        "target_flag": "test",
        "target_reverse": True,
        "title": "Head-direction reversal reconstruction",
        "suptitle": "RatInABox head-direction reversal with held-out speed",
        "stem_prefix": "inverse_trajectory_",
    },
}


def configure_tensorflow_runtime(tf_module):
    try:
        for gpu in tf_module.config.list_physical_devices("GPU"):
            tf_module.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # TensorFlow raises if the runtime is already initialized; training can
        # still proceed, but memory growth may not be applied for that process.
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433

        configure_tensorflow_runtime(tf)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def make_loader(dataset, batch_size: int, shuffle: bool = False):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def collate_behaviour_and_target(dataloader, target_key: str):
    batch = collate_all_batches(dataloader)
    return collate_batch_behaviour_and_target(batch, target_key)


def collate_batch_behaviour_and_target(batch, target_key: str):
    x = torch.cat([batch["pos"], batch["head_direction"]], dim=1).float()
    y = batch[target_key].squeeze().float()
    return x, y


def collate_display_rates(dataloader):
    batch = collate_all_batches(dataloader)
    return collate_batch_display_rates(batch)


def collate_batch_display_rates(batch):
    return batch["rates"].squeeze().float()


class DictBatchDataset(torch.utils.data.Dataset):
    def __init__(self, batch):
        self.batch = batch
        self.length = int(next(iter(batch.values())).shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.batch.items()}


def make_dict_loader(batch, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset=DictBatchDataset(batch),
        batch_size=batch_size,
        shuffle=False,
    )


def _crop_batch(batch, n_samples: int):
    return {
        key: value[:n_samples] if torch.is_tensor(value) and value.shape[0] >= n_samples else value
        for key, value in batch.items()
    }


def _slice_batch(batch, start=None, stop=None):
    n_samples = int(batch["spikes"].shape[0])
    return {
        key: value[start:stop] if torch.is_tensor(value) and value.shape[0] == n_samples else value
        for key, value in batch.items()
    }


def _reverse_batch(batch):
    n_samples = int(batch["spikes"].shape[0])
    reversed_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] == n_samples:
            reversed_batch[key] = value.flip(dims=(0,))
        else:
            reversed_batch[key] = value
    return reversed_batch


def _align_reversed_windowed_batches(source, target, datamodule):
    target_mode = (
        getattr(datamodule.hparams, "behaviour_target_mode", None)
        or getattr(datamodule.hparams, "spike_target_mode", None)
    )
    if target_mode != "input_trailing":
        return source, target

    lag = int(getattr(datamodule.hparams, "input_window", 1)) - int(
        getattr(datamodule.hparams, "output_window", 1)
    )
    if lag <= 0:
        return source, target

    # The inverse trajectory target is reversed after windowing. For trailing
    # targets, this shifts target behaviours/rates earlier by input_window - 1
    # bins; crop both batches so source and target refer to the same raw times.
    return _slice_batch(source, None, -lag), _slice_batch(target, lag, None)


def _collate_dataset(dataset, batch_size: int):
    return collate_all_batches(make_loader(dataset, batch_size, shuffle=False))


def make_counterfactual_batches(datamodule, batch_size: int, counterfactual_mode: str):
    if counterfactual_mode not in COUNTERFACTUAL_MODES:
        raise ValueError(
            f"Unknown counterfactual mode {counterfactual_mode!r}. "
            f"Use one of: {', '.join(sorted(COUNTERFACTUAL_MODES))}."
        )
    mode = COUNTERFACTUAL_MODES[counterfactual_mode]
    source = _collate_dataset(datamodule.dataset(mode["source_flag"]), batch_size)
    target = _collate_dataset(datamodule.dataset(mode["target_flag"]), batch_size)
    if mode["target_reverse"]:
        target = _reverse_batch(target)
        source, target = _align_reversed_windowed_batches(source, target, datamodule)
    n_samples = min(int(source["spikes"].shape[0]), int(target["spikes"].shape[0]))
    return {
        "source": _crop_batch(source, n_samples),
        "target": _crop_batch(target, n_samples),
        "mode": mode,
    }


class PIVAEBehaviourOnly:
    def __init__(self, vae, latent_dim: int):
        self.vae = vae
        self.encoder = vae.get_layer("encoder")
        self.decoder = vae.get_layer("decoder")
        self.latent_dim = latent_dim


def _import_pivae():
    pivae_code_dir = ROOT / "pi-vae" / "code"
    if not pivae_code_dir.exists():
        raise FileNotFoundError(
            f"Expected pi-VAE checkout at {pivae_code_dir}. Clone https://github.com/zhd96/pi-vae first."
        )
    if str(pivae_code_dir) not in sys.path:
        sys.path.append(str(pivae_code_dir))
    try:
        import types
        import tensorflow as tf  # noqa: WPS433

        configure_tensorflow_runtime(tf)
        if not hasattr(tf, "log"):
            tf.log = tf.math.log
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # noqa: WPS433

        source_path = pivae_code_dir / "pi_vae.py"
        source = source_path.read_text(encoding="utf-8")
        replacements = {
            "from keras.models import Sequential": "from tensorflow.keras.models import Sequential",
            "from keras.optimizers import Adam": "from tensorflow.keras.optimizers import Adam",
            "from keras import layers": "from tensorflow.keras import layers",
            "from keras.models import Model": "from tensorflow.keras.models import Model",
            "from keras import losses": "from tensorflow.keras import losses",
            "from keras.layers.core import Lambda": "from tensorflow.keras.layers import Lambda",
            "from keras import backend as K": "from tensorflow.keras import backend as K",
            "from keras.callbacks import ModelCheckpoint": "from tensorflow.keras.callbacks import ModelCheckpoint",
            "from keras.activations import softplus, sigmoid": "from tensorflow.keras.activations import softplus, sigmoid",
            "from keras.callbacks import LearningRateScheduler": "from tensorflow.keras.callbacks import LearningRateScheduler",
            "Adam(lr = learning_rate)": "Adam(learning_rate=learning_rate)",
            "print(vae.summary())": "",
            "tf.log(": "tf.math.log(",
        }
        for old, new in replacements.items():
            source = source.replace(old, new)

        module = types.ModuleType("pi_vae_compat")
        module.__file__ = str(source_path)
        exec(compile(source, str(source_path), "exec"), module.__dict__)
        vae_mdl = module.vae_mdl
    except Exception as exc:
        raise RuntimeError(
            "pi-VAE import failed. Ensure tensorflow/keras are installed in the active environment."
        ) from exc
    return vae_mdl, EarlyStopping, ModelCheckpoint


def _as_numpy(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _flatten_activity(activity):
    activity = _as_numpy(activity)
    if activity.ndim <= 2:
        return activity
    return activity.reshape(activity.shape[0], -1)


def _smooth_display_path(path):
    path = np.asarray(path, dtype=float)
    if len(path) < 5:
        return path.copy()
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(path, ((2, 2), (0, 0)), mode="edge")
    smoothed = np.empty_like(path, dtype=float)
    for dim in range(path.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def _shared_xy_limits(paths, pad_fraction=0.03, min_pad=0.02):
    stacked = np.concatenate(paths, axis=0)
    x_low, x_high = np.percentile(stacked[:, 0], [1.0, 99.0])
    y_low, y_high = np.percentile(stacked[:, 1], [1.0, 99.0])
    x_pad = max(min_pad, pad_fraction * max(x_high - x_low, min_pad))
    y_pad = max(min_pad, pad_fraction * max(y_high - y_low, min_pad))
    return (x_low - x_pad, x_high + x_pad), (y_low - y_pad, y_high + y_pad)


def _cebra_decoder_kwargs(args):
    return {
        "seed": args.seed,
        "cebra_model_architecture": args.cebra_model_architecture,
        "cebra_batch_size": args.cebra_batch_size,
        "cebra_learning_rate": args.cebra_learning_rate,
        "cebra_temperature": args.cebra_temperature,
        "cebra_output_dimension": args.cebra_output_dimension,
        "cebra_max_iterations": args.cebra_max_iterations,
        "cebra_distance": args.cebra_distance,
        "cebra_conditional": args.cebra_conditional,
        "cebra_time_offsets": args.cebra_time_offsets,
        "cebra_knn_neighbors": args.cebra_knn_neighbors,
        "cebra_knn_metric": args.cebra_knn_metric,
        "cebra_verbose": bool(args.cebra_verbose),
    }


def fit_pivae_baseline(train_u, train_x, val_u, val_x, args, checkpoint_path: Path):
    vae_mdl, EarlyStopping, ModelCheckpoint = _import_pivae()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    train_u = _as_numpy(train_u)
    train_x = _as_numpy(train_x)
    val_u = _as_numpy(val_u)
    val_x = _as_numpy(val_x)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    n_flow_blocks = args.pivae_flow_blocks if args.pivae_flow_blocks > 0 else None
    vae = vae_mdl(
        dim_x=train_x.shape[1],
        dim_z=args.pivae_latent_dim,
        dim_u=train_u.shape[1],
        gen_nodes=args.pivae_gen_nodes,
        n_blk=n_flow_blocks,
        mdl="poisson",
        disc=False,
        learning_rate=args.pivae_lr,
    )

    if checkpoint_path.exists() and not args.force_baseline_retrain:
        vae.load_weights(str(checkpoint_path))
        model = PIVAEBehaviourOnly(vae=vae, latent_dim=args.pivae_latent_dim)
        return model, {
            "reused_checkpoint": True,
            "checkpoint_path": str(checkpoint_path),
        }

    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.pivae_patience,
            min_delta=1e-4,
            restore_best_weights=False,
        ),
    ]

    history = vae.fit(
        [train_x, train_u],
        None,
        validation_data=([val_x, val_u], None),
        epochs=args.pivae_epochs,
        batch_size=args.pivae_batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )

    if checkpoint_path.exists():
        vae.load_weights(str(checkpoint_path))

    val_losses = [float(v) for v in history.history.get("val_loss", [])]
    best_epoch = int(np.argmin(val_losses)) if val_losses else -1
    best_val_loss = float(np.min(val_losses)) if val_losses else float("nan")
    training_history = []
    train_losses = history.history.get("loss", [])
    for idx in range(len(train_losses)):
        row = {"epoch": idx, "train_loss": float(train_losses[idx])}
        if idx < len(val_losses):
            row["val_loss"] = float(val_losses[idx])
        training_history.append(row)

    model = PIVAEBehaviourOnly(vae=vae, latent_dim=args.pivae_latent_dim)
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": training_history,
    }


def _decoder_predict_array(decoder, z, batch_size):
    pred = decoder.predict(z, batch_size=batch_size, verbose=0)
    if isinstance(pred, list):
        pred = pred[0]
    return _as_numpy(pred)


def predict_pivae(model: PIVAEBehaviourOnly, u, x_dim: int, batch_size: int, n_samples: int):
    u = _as_numpy(u)
    dummy_x = np.zeros((u.shape[0], x_dim), dtype=np.float32)
    enc_out = model.encoder.predict([dummy_x, u], batch_size=batch_size, verbose=0)
    lam_mean = _as_numpy(enc_out[5])
    lam_log_var = _as_numpy(enc_out[6])

    if n_samples <= 1:
        return _decoder_predict_array(model.decoder, lam_mean, batch_size=batch_size)

    z = np.random.normal(size=(n_samples, lam_mean.shape[0], model.latent_dim)).astype(np.float32)
    z *= np.exp(0.5 * lam_log_var)[None, :, :]
    z += lam_mean[None, :, :]
    pred = _decoder_predict_array(model.decoder, z.reshape(-1, model.latent_dim), batch_size=batch_size)
    return pred.reshape(n_samples, lam_mean.shape[0], -1).mean(axis=0)


def build_counterfactual_prediction(main_model, cfg, source_loader, target_loader, target_key: str):
    source_batch = collate_all_batches(source_loader)
    target_batch = collate_all_batches(target_loader)
    unknown_input_key = getattr(cfg, "unknown_input_key", "spikes")

    cf_batch = {
        "spikes": source_batch["spikes"],
        unknown_input_key: source_batch[unknown_input_key],
    }
    if "spikes_context" in source_batch:
        cf_batch["spikes_context"] = source_batch["spikes_context"]
    for behaviour in main_model.behaviours:
        cf_batch[behaviour] = target_batch[behaviour]

    with torch.no_grad():
        pred = main_model(cf_batch)["reconstruction"].squeeze().cpu().numpy()
    true = target_batch[target_key].squeeze().cpu().numpy()
    return pred, true


def predict_main_model(main_model, dataloader, target_key: str):
    batch = collate_all_batches(dataloader)
    with torch.no_grad():
        pred = main_model(batch)["reconstruction"].squeeze().cpu().numpy()
    true = batch[target_key].squeeze().cpu().numpy()
    return pred, true


def group_slices(cfg):
    group_names = ["Place Cells", "Head Direction Cells", "Speed Cells"]
    if getattr(cfg, "include_noise_cell", False):
        group_names.append("Noise Cells")
    slices = {}
    for idx, name in enumerate(group_names):
        start = idx * cfg.num_cells
        slices[name] = slice(start, start + cfg.num_cells)
    slices["All Cells"] = slice(0, cfg.num_neurons)
    return slices


def compute_group_r2(y_true, y_pred, cfg):
    return {name: float(r2_score(y_true[:, s], y_pred[:, s])) for name, s in group_slices(cfg).items()}


def bits_per_spike(pred_rates: np.ndarray, spikes: np.ndarray) -> float:
    eps = 1e-8
    pred_rates = np.clip(np.asarray(pred_rates, dtype=np.float32), eps, None)
    spikes = np.asarray(spikes, dtype=np.float32)
    null_rates = np.clip(spikes.mean(axis=0, keepdims=True), eps, None)
    loglik_model = (spikes * np.log(pred_rates) - pred_rates).sum()
    loglik_null = (spikes * np.log(null_rates) - null_rates).sum()
    denom = max(float(spikes.sum()) * np.log(2.0), eps)
    return float((loglik_model - loglik_null) / denom)


def compute_group_bps(target_spikes, pred_rates, cfg):
    return {
        name: bits_per_spike(pred_rates[:, s], target_spikes[:, s])
        for name, s in group_slices(cfg).items()
    }


def _target_display_name(target_key: str) -> str:
    return "spikes" if target_key == "spikes" else "rates"


def _figure_stem(target_key: str, counterfactual_mode: str, behaviour_preservation: bool) -> str:
    if target_key == "rates":
        stem = "behaviour_only_baseline_vs_cf"
    else:
        stem = f"behaviour_only_baseline_vs_cf_on_{target_key}"
    stem = f"{COUNTERFACTUAL_MODES[counterfactual_mode]['stem_prefix']}{stem}"
    if behaviour_preservation:
        stem = f"{stem}_behaviour_preservation"
    return stem


def _summary_metrics(target_key: str, target_activity, baseline_pred, cf_pred, cfg, target_rates=None):
    if target_key == "spikes":
        # For spike-trained models we still predict rates; when GT rates exist,
        # compare rate predictions directly to the GT rates for figure summaries.
        if target_rates is not None:
            return (
                compute_group_r2(target_rates, baseline_pred, cfg),
                compute_group_r2(target_rates, cf_pred, cfg),
                "r2",
            )
        return (
            compute_group_bps(target_activity, baseline_pred, cfg),
            compute_group_bps(target_activity, cf_pred, cfg),
            "bps",
        )
    return (
        compute_group_r2(target_activity, baseline_pred, cfg),
        compute_group_r2(target_activity, cf_pred, cfg),
        "r2",
    )


def _metric_label(metric_key: str) -> str:
    return "BPS" if metric_key == "bps" else "R²"


PLOT_GROUP_ALIASES = {
    "place": "Place Cells",
    "position": "Place Cells",
    "head": "Head Direction Cells",
    "head_direction": "Head Direction Cells",
    "speed": "Speed Cells",
    "all": "All Cells",
}


def _resolve_plot_groups(names):
    if not names:
        return ["Place Cells", "Head Direction Cells", "Speed Cells", "All Cells"]
    groups = []
    for name in names:
        key = name.strip().lower().replace("-", "_")
        if key not in PLOT_GROUP_ALIASES:
            raise ValueError(
                f"Unknown plot group {name!r}. Use one of: {', '.join(sorted(PLOT_GROUP_ALIASES))}."
            )
        group = PLOT_GROUP_ALIASES[key]
        if group not in groups:
            groups.append(group)
    return groups


def _short_group_label(group):
    return {
        "Place Cells": "Place",
        "Head Direction Cells": "Head dir",
        "Speed Cells": "Speed",
        "All Cells": "All",
    }[group]


def _split_figure_paths(save_path):
    save_path = Path(save_path)
    return {
        "combined": str(save_path),
        "traces": str(save_path.with_name(f"{save_path.stem}_traces{save_path.suffix}")),
        "reconstruction": str(save_path.with_name(f"{save_path.stem}_reconstruction{save_path.suffix}")),
        "reconstruction_speed": str(
            save_path.with_name(f"{save_path.stem}_reconstruction_speed_only{save_path.suffix}")
        ),
        "trajectory": str(save_path.with_name(f"{save_path.stem}_trajectory{save_path.suffix}")),
    }


def _plot_decoded_trajectory_panel(
    arrays,
    save_path,
    display_points,
    title,
):
    n_points = min(display_points, *(len(value) for value in arrays.values()))
    paths = {
        key: _smooth_display_path(_as_numpy(value)[:n_points].reshape(n_points, -1)[:, :2])
        for key, value in arrays.items()
    }
    xlim, ylim = _shared_xy_limits(list(paths.values()))

    fig, ax = plt.subplots(figsize=(7.0, 3.65))
    traces = [
        ("source_true", "Source", COUNTERFACTUAL_COLORS["source_true"], "-", 1.8, 0.90),
        ("source_decoded", "Source decode", COUNTERFACTUAL_COLORS["source_decode"], "-", 2.0, 0.98),
        ("target_true", "Target", COUNTERFACTUAL_COLORS["target_true"], "-", 1.8, 0.96),
        ("cf_decoded", "Neural-DISCO", COUNTERFACTUAL_COLORS["counterfactual"], "--", 1.9, 0.98),
    ]
    for key, label, color, linestyle, linewidth, alpha in traces:
        path = paths[key]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12, pad=34)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(color=COUNTERFACTUAL_COLORS["grid"], linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.78)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_decoded_position_trajectory(cfg, args, log_dir, device, fold_idx=None):
    model_log_dir = log_dir if fold_idx is None else f"{log_dir}/fold_{fold_idx}"
    main_model = _load_main_model_for_log(cfg, model_log_dir)
    datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx or 0)
    train_dataset = datamodule.dataset("train") if fold_idx is None else datamodule.train_dataset
    train_batch = _collate_dataset(train_dataset, args.batch_size)
    counterfactual_batches = make_counterfactual_batches(
        datamodule,
        args.batch_size,
        args.counterfactual_mode,
    )
    source_loader = make_dict_loader(counterfactual_batches["source"], args.batch_size)
    target_loader = make_dict_loader(counterfactual_batches["target"], args.batch_size)
    target_key = getattr(cfg, "reconstruction_target_key", "rates")
    cf_pred, _ = build_counterfactual_prediction(
        main_model, cfg, source_loader, target_loader, target_key
    )

    from scripts.cfact.mc_maze_mlp_decode_xy import decoder_predict, fit_decoder

    if args.trajectory_decode_activity == "auto":
        activity_key = "rates" if "rates" in train_batch and "rates" in counterfactual_batches["source"] else target_key
    else:
        activity_key = args.trajectory_decode_activity
    if activity_key not in train_batch or activity_key not in counterfactual_batches["source"]:
        raise KeyError(f"Trajectory decode activity key {activity_key!r} is missing from RatInABox batches.")
    train_activity = _flatten_activity(train_batch[activity_key])
    source_activity = _flatten_activity(counterfactual_batches["source"][activity_key])
    decoder = fit_decoder(
        train_activity,
        _as_numpy(train_batch["pos"]),
        decoder_type="cebra",
        decoder_kwargs=_cebra_decoder_kwargs(args),
    )
    return {
        "source_true": _as_numpy(counterfactual_batches["source"]["pos"]),
        "source_decoded": decoder_predict(decoder, source_activity),
        "target_true": _as_numpy(counterfactual_batches["target"]["pos"]),
        "cf_decoded": decoder_predict(decoder, _flatten_activity(cf_pred)),
    }


def _plot_trace_panel(
    display_target,
    baseline_pred,
    cf_pred,
    ablation_cf_pred,
    cfg,
    trace_groups,
    time,
    slices,
    save_path,
    y_label,
    trace_label,
    baseline_label,
    cf_label,
    ablation_label,
    show_ablation_traces,
):
    fig, axes = plt.subplots(
        len(trace_groups),
        1,
        figsize=(7.0, 2.98),
        sharex=True,
        constrained_layout=True,
    )
    if len(trace_groups) == 1:
        axes = [axes]

    for row, (ax, group) in enumerate(zip(axes, trace_groups)):
        sample_slice = slices[group]
        sample_idx = sample_slice.start + cfg.num_cells // 2
        ax.plot(
            time,
            display_target[: len(time), sample_idx],
            color=COUNTERFACTUAL_COLORS["target_true"],
            linewidth=1.25,
            label=trace_label,
        )
        ax.plot(
            time,
            baseline_pred[: len(time), sample_idx],
            color=COUNTERFACTUAL_SUMMARY_COLORS["baseline"],
            linewidth=1.25,
            label=baseline_label,
        )
        if ablation_cf_pred is not None and show_ablation_traces:
            ax.plot(
                time,
                ablation_cf_pred[: len(time), sample_idx],
                color=COUNTERFACTUAL_SUMMARY_COLORS["ablation"],
                linestyle="--",
                linewidth=1.1,
                label=ablation_label,
            )
        ax.plot(
            time,
            cf_pred[: len(time), sample_idx],
            color=COUNTERFACTUAL_COLORS["counterfactual"],
            linewidth=1.25,
            label=cf_label,
        )
        ax.set_title(group, fontsize=7.5, pad=1.5)
        ax.set_ylabel(y_label, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=COUNTERFACTUAL_COLORS["grid"], linewidth=0.6, alpha=0.55)
        ax.tick_params(axis="both", labelsize=7)
        if row == len(trace_groups) - 1:
            ax.set_xlabel("Time", fontsize=8)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _draw_reconstruction_bar_panel(
    categories,
    baseline_metrics,
    cf_metrics,
    ablation_metrics,
    baseline_sem,
    cf_sem,
    ablation_sem,
    metric_key,
    target_key,
    target_name,
    baseline_label,
    cf_label,
    ablation_label,
    clip_ablation_bars,
    plot_title,
    save_path,
    figsize=None,
):
    if not categories:
        raise ValueError("Cannot plot an empty reconstruction category list.")
    x = np.arange(len(categories))
    has_ablation = ablation_metrics is not None
    width = 0.23 if has_ablation else 0.34
    baseline_values = np.array([baseline_metrics[name] for name in categories], dtype=float)
    cf_values = np.array([cf_metrics[name] for name in categories], dtype=float)
    ablation_values = (
        np.array([ablation_metrics[name] for name in categories], dtype=float) if has_ablation else None
    )
    baseline_yerr = None if baseline_sem is None else np.array([baseline_sem[name] for name in categories], dtype=float)
    cf_yerr = None if cf_sem is None else np.array([cf_sem[name] for name in categories], dtype=float)
    ablation_yerr = (
        None if ablation_sem is None else np.array([ablation_sem[name] for name in categories], dtype=float)
    )

    clip_floor = -0.12 if metric_key == "r2" else None
    clipped_ablation = (
        has_ablation
        and clip_ablation_bars
        and clip_floor is not None
        and np.any(ablation_values < clip_floor)
    )
    ablation_plot_values = (
        np.maximum(ablation_values, clip_floor) if clipped_ablation else ablation_values
    )
    ablation_plot_yerr = None if clipped_ablation else ablation_yerr
    error_kw = {"elinewidth": 1.1, "capthick": 1.1, "capsize": 2.5}

    if figsize is None:
        figsize = (max(3.5, 2.2 * len(categories) + 1.4), 4.45)
    fig, ax_bar = plt.subplots(figsize=figsize, constrained_layout=True)
    ablation_x = x - width
    baseline_x = x if has_ablation else x - width / 2
    cf_x = x + width if has_ablation else x + width / 2
    if has_ablation:
        ax_bar.bar(
            ablation_x,
            ablation_plot_values,
            width,
            yerr=ablation_plot_yerr,
            error_kw=error_kw,
            color=COUNTERFACTUAL_SUMMARY_COLORS["ablation"],
            edgecolor="#555555",
            hatch="///",
            linewidth=0.55,
            label=f"{ablation_label} (clipped)" if clipped_ablation else ablation_label,
        )
    ax_bar.bar(
        baseline_x,
        baseline_values,
        width,
        yerr=baseline_yerr,
        error_kw=error_kw,
        color=COUNTERFACTUAL_SUMMARY_COLORS["baseline"],
        label=baseline_label,
    )
    ax_bar.bar(
        cf_x,
        cf_values,
        width,
        yerr=cf_yerr,
        error_kw=error_kw,
        color=COUNTERFACTUAL_COLORS["counterfactual"],
        label=cf_label,
    )
    ax_bar.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([_short_group_label(group) for group in categories])
    ax_bar.set_xlim(-0.55, len(categories) - 0.45)

    visible_values = np.concatenate([baseline_values, cf_values])
    if not clipped_ablation and has_ablation:
        visible_values = np.concatenate([visible_values, ablation_values])
    y_lower = min(-0.05, float(np.nanmin(visible_values)) - 0.05)
    y_upper = max(1.0, float(np.nanmax(visible_values)) + 0.05)
    if has_ablation:
        y_upper = max(y_upper, 1.18)
    if clipped_ablation:
        y_lower = min(y_lower, clip_floor - 0.08)
        for xpos, raw_value, plotted_value, sem in zip(
            ablation_x,
            ablation_values,
            ablation_plot_values,
            np.zeros_like(ablation_values) if ablation_yerr is None else ablation_yerr,
        ):
            if raw_value < clip_floor:
                ax_bar.plot(xpos, plotted_value, marker="v", color="#555555", markersize=5, clip_on=False)
                if abs(raw_value) >= 10:
                    label = f"{raw_value:.0f}"
                    if ablation_yerr is not None:
                        label = f"{raw_value:.0f}±{sem:.0f}"
                else:
                    label = f"{raw_value:.2f}"
                    if ablation_yerr is not None:
                        label = f"{raw_value:.2f}±{sem:.2f}"
                ax_bar.annotate(
                    label,
                    xy=(xpos, plotted_value),
                    xytext=(0, -9),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="#444444",
                    clip_on=False,
                )
    ax_bar.set_ylim(y_lower, y_upper)
    if target_key == "spikes":
        ax_bar.set_ylabel("Bits per spike to target spikes" if metric_key == "bps" else "R² to target rates")
    else:
        ax_bar.set_ylabel(f"R² to target {target_name}")
    ax_bar.set_title(plot_title, fontsize=12, pad=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", color=COUNTERFACTUAL_COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax_bar.tick_params(axis="both", labelsize=9)
    ax_bar.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reconstruction_panel(
    categories,
    baseline_metrics,
    cf_metrics,
    ablation_metrics,
    baseline_sem,
    cf_sem,
    ablation_sem,
    metric_key,
    target_key,
    target_name,
    baseline_label,
    cf_label,
    ablation_label,
    clip_ablation_bars,
    plot_title,
    save_path,
):
    save_path = Path(save_path)
    categories = [category for category in categories if category != "All Cells"]
    if not categories:
        categories = ["All Cells"]

    speed_categories = [category for category in categories if category == "Speed Cells"]
    main_categories = [category for category in categories if category != "Speed Cells"]
    output_paths = {"reconstruction": str(save_path)}

    if main_categories:
        _draw_reconstruction_bar_panel(
            categories=main_categories,
            baseline_metrics=baseline_metrics,
            cf_metrics=cf_metrics,
            ablation_metrics=ablation_metrics,
            baseline_sem=baseline_sem,
            cf_sem=cf_sem,
            ablation_sem=ablation_sem,
            metric_key=metric_key,
            target_key=target_key,
            target_name=target_name,
            baseline_label=baseline_label,
            cf_label=cf_label,
            ablation_label=ablation_label,
            clip_ablation_bars=clip_ablation_bars,
            plot_title=plot_title,
            save_path=save_path,
        )

    if speed_categories:
        speed_path = (
            save_path
            if not main_categories
            else save_path.with_name(f"{save_path.stem}_speed_only{save_path.suffix}")
        )
        _draw_reconstruction_bar_panel(
            categories=speed_categories,
            baseline_metrics=baseline_metrics,
            cf_metrics=cf_metrics,
            ablation_metrics=ablation_metrics,
            baseline_sem=baseline_sem,
            cf_sem=cf_sem,
            ablation_sem=ablation_sem,
            metric_key=metric_key,
            target_key=target_key,
            target_name=target_name,
            baseline_label=baseline_label,
            cf_label=cf_label,
            ablation_label=ablation_label,
            clip_ablation_bars=clip_ablation_bars,
            plot_title="Behaviour Preservation",
            save_path=speed_path,
            figsize=(3.6, 4.45),
        )
        output_paths["reconstruction_speed"] = str(speed_path)

    return output_paths


def _write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _export_plot_data(
    data_export_prefix,
    display_target,
    baseline_pred,
    cf_pred,
    ablation_cf_pred,
    cfg,
    trace_groups,
    time,
    slices,
    trace_label,
    baseline_label,
    cf_label,
    ablation_label,
    trajectory_arrays,
    display_points,
):
    if data_export_prefix is None:
        return {}

    prefix = Path(data_export_prefix)
    trace_rows = []
    series = [
        ("target", trace_label, display_target),
        ("baseline", baseline_label, baseline_pred),
        ("counterfactual", cf_label, cf_pred),
    ]
    if ablation_cf_pred is not None:
        series.append(("ablation", ablation_label, ablation_cf_pred))

    for group in trace_groups:
        sample_slice = slices[group]
        sample_idx = sample_slice.start + cfg.num_cells // 2
        for series_key, series_label, values in series:
            values = np.asarray(values)
            for time_idx in time:
                trace_rows.append(
                    {
                        "cell_group": group,
                        "cell_index": int(sample_idx),
                        "time_index": int(time_idx),
                        "series": series_key,
                        "series_label": series_label,
                        "value": float(values[int(time_idx), sample_idx]),
                    }
                )

    data_paths = {
        "traces": _write_csv_rows(
            prefix.with_name(f"{prefix.name}_traces.csv"),
            ["cell_group", "cell_index", "time_index", "series", "series_label", "value"],
            trace_rows,
        )
    }

    if trajectory_arrays is not None:
        trajectory_rows = []
        trajectory_series = [
            ("source_true", "Source", trajectory_arrays["source_true"]),
            ("source_decoded", "Source decode", trajectory_arrays["source_decoded"]),
            ("target_true", "Target", trajectory_arrays["target_true"]),
            ("cf_decoded", cf_label, trajectory_arrays["cf_decoded"]),
        ]
        n_points = min(display_points, *(len(value) for _, _, value in trajectory_series))
        for series_key, series_label, values in trajectory_series:
            values = np.asarray(values)[:n_points].reshape(n_points, -1)[:, :2]
            for time_idx, xy in enumerate(values):
                trajectory_rows.append(
                    {
                        "time_index": int(time_idx),
                        "series": series_key,
                        "series_label": series_label,
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                    }
                )
        data_paths["trajectory"] = _write_csv_rows(
            prefix.with_name(f"{prefix.name}_trajectory.csv"),
            ["time_index", "series", "series_label", "x", "y"],
            trajectory_rows,
        )

    return data_paths


def plot_comparison(
    target_activity,
    baseline_pred,
    cf_pred,
    cfg,
    save_path,
    display_points,
    target_key: str,
    display_rates=None,
    metric_overrides=None,
    ablation_cf_pred=None,
    baseline_label="pi-VAE",
    cf_label="Neural-DISCO",
    ablation_label="w/o reg.",
    show_ablation_traces=False,
    clip_ablation_bars=True,
    plot_groups=None,
    plot_title="Position-shift reconstruction",
    suptitle="RatInABox position shift with held-out speed",
    trajectory_arrays=None,
    data_export_prefix=None,
):
    slices = group_slices(cfg)
    categories = _resolve_plot_groups(plot_groups)
    bar_categories = [group for group in categories if group != "All Cells"] or categories
    trace_groups = [group for group in bar_categories if group != "All Cells"]
    target_name = _target_display_name(target_key)

    if metric_overrides is None:
        baseline_metrics, cf_metrics, metric_key = _summary_metrics(
            target_key, target_activity, baseline_pred, cf_pred, cfg, target_rates=display_rates
        )
        ablation_metrics = None
        if ablation_cf_pred is not None:
            if target_key == "spikes" and display_rates is not None:
                ablation_metrics = compute_group_r2(display_rates, ablation_cf_pred, cfg)
            elif target_key == "spikes":
                ablation_metrics = compute_group_bps(target_activity, ablation_cf_pred, cfg)
            else:
                ablation_metrics = compute_group_r2(target_activity, ablation_cf_pred, cfg)
        baseline_sem = None
        cf_sem = None
        ablation_sem = None
    else:
        baseline_metrics = metric_overrides["baseline_mean"]
        cf_metrics = metric_overrides["counterfactual_mean"]
        ablation_metrics = metric_overrides.get("ablation_counterfactual_mean")
        metric_key = metric_overrides["metric_key"]
        baseline_sem = metric_overrides.get("baseline_sem")
        cf_sem = metric_overrides.get("counterfactual_sem")
        ablation_sem = metric_overrides.get("ablation_counterfactual_sem")
    metric_label = _metric_label(metric_key)

    fig_height = max(5.4, 2.55 * len(trace_groups) + 0.9)
    fig = plt.figure(figsize=(12.4, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(len(trace_groups), 2, width_ratios=[2.15, 1.0])

    time = np.arange(min(display_points, len(target_activity)))
    if display_rates is None:
        display_target = target_activity
        trace_label = f"Target {target_name}"
        y_label = "Activity"
    else:
        display_target = display_rates
        trace_label = "Target rates"
        y_label = "Rate"
    for row, group in enumerate(trace_groups):
        ax = fig.add_subplot(grid[row, 0])
        sample_slice = slices[group]
        sample_idx = sample_slice.start + cfg.num_cells // 2
        ax.plot(
            time,
            display_target[: len(time), sample_idx],
            color=COUNTERFACTUAL_COLORS["target_true"],
            label=trace_label,
        )
        ax.plot(
            time,
            baseline_pred[: len(time), sample_idx],
            color=COUNTERFACTUAL_SUMMARY_COLORS["baseline"],
            label=baseline_label,
        )
        if ablation_cf_pred is not None and show_ablation_traces:
            ax.plot(
                time,
                ablation_cf_pred[: len(time), sample_idx],
                color=COUNTERFACTUAL_SUMMARY_COLORS["ablation"],
                linestyle="--",
                linewidth=1.5,
                label=ablation_label,
            )
        ax.plot(time, cf_pred[: len(time), sample_idx], color=COUNTERFACTUAL_COLORS["counterfactual"], label=cf_label)
        ax.set_title(
            f"{group}: {baseline_label} {metric_label}={baseline_metrics[group]:.3f}, {cf_label} {metric_label}={cf_metrics[group]:.3f}",
            fontsize=12,
        )
        ax.set_ylabel(y_label)
        if row == 0:
            ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
        if row == len(trace_groups) - 1:
            ax.set_xlabel("Time")

    ax_bar = fig.add_subplot(grid[:, 1])
    x = np.arange(len(bar_categories))
    has_ablation = ablation_metrics is not None
    width = 0.25 if has_ablation else 0.35
    baseline_values = np.array([baseline_metrics[name] for name in bar_categories], dtype=float)
    cf_values = np.array([cf_metrics[name] for name in bar_categories], dtype=float)
    ablation_values = (
        np.array([ablation_metrics[name] for name in bar_categories], dtype=float) if has_ablation else None
    )
    baseline_yerr = (
        None if baseline_sem is None else np.array([baseline_sem[name] for name in bar_categories], dtype=float)
    )
    cf_yerr = None if cf_sem is None else np.array([cf_sem[name] for name in bar_categories], dtype=float)
    ablation_yerr = (
        None if ablation_sem is None else np.array([ablation_sem[name] for name in bar_categories], dtype=float)
    )
    clip_floor = -0.12 if metric_key == "r2" else None
    clipped_ablation = (
        has_ablation
        and clip_ablation_bars
        and clip_floor is not None
        and np.any(ablation_values < clip_floor)
    )
    ablation_plot_values = (
        np.maximum(ablation_values, clip_floor) if clipped_ablation else ablation_values
    )
    ablation_plot_yerr = None if clipped_ablation else ablation_yerr
    error_kw = {"elinewidth": 1.2, "capthick": 1.2, "capsize": 3}
    ablation_x = x - width
    baseline_x = x if has_ablation else x - width / 2
    cf_x = x + width if has_ablation else x + width / 2
    if has_ablation:
        ax_bar.bar(
            ablation_x,
            ablation_plot_values,
            width,
            yerr=ablation_plot_yerr,
            error_kw=error_kw,
            color=COUNTERFACTUAL_SUMMARY_COLORS["ablation"],
            edgecolor="#555555",
            hatch="///",
            linewidth=0.6,
            label=f"{ablation_label} (clipped)" if clipped_ablation else ablation_label,
        )
    ax_bar.bar(
        baseline_x,
        baseline_values,
        width,
        yerr=baseline_yerr,
        error_kw=error_kw,
        color=COUNTERFACTUAL_SUMMARY_COLORS["baseline"],
        label=baseline_label,
    )
    ax_bar.bar(
        cf_x,
        cf_values,
        width,
        yerr=cf_yerr,
        error_kw=error_kw,
        color=COUNTERFACTUAL_COLORS["counterfactual"],
        label=cf_label,
    )
    ax_bar.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([_short_group_label(group) for group in bar_categories])
    visible_values = np.concatenate([baseline_values, cf_values])
    if not clipped_ablation and has_ablation:
        visible_values = np.concatenate([visible_values, ablation_values])
    y_lower = min(-0.05, float(np.nanmin(visible_values)) - 0.05)
    y_upper = max(1.0, float(np.nanmax(visible_values)) + 0.05)
    if has_ablation:
        y_upper = max(y_upper, 1.18)
    if clipped_ablation:
        y_lower = min(y_lower, clip_floor - 0.08)
        for xpos, raw_value, plotted_value, sem in zip(
            ablation_x,
            ablation_values,
            ablation_plot_values,
            np.zeros_like(ablation_values) if ablation_yerr is None else ablation_yerr,
        ):
            if raw_value < clip_floor:
                ax_bar.plot(xpos, plotted_value, marker="v", color="#555555", markersize=5, clip_on=False)
                if abs(raw_value) >= 10:
                    label = f"{raw_value:.0f}"
                    if ablation_yerr is not None:
                        label = f"{raw_value:.0f}±{sem:.0f}"
                else:
                    label = f"{raw_value:.2f}"
                    if ablation_yerr is not None:
                        label = f"{raw_value:.2f}±{sem:.2f}"
                ax_bar.annotate(
                    label,
                    xy=(xpos, plotted_value),
                    xytext=(0, -13),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="#444444",
                    clip_on=False,
                )
    ax_bar.set_ylim(y_lower, y_upper)
    if target_key == "spikes":
        if metric_key == "bps":
            ax_bar.set_ylabel("Bits per spike to target spikes")
        else:
            ax_bar.set_ylabel("R² to target rates")
    else:
        ax_bar.set_ylabel(f"R² to target {target_name}")
    ax_bar.set_title(plot_title)
    ax_bar.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle(suptitle, fontsize=14)
    figure_paths = _split_figure_paths(save_path)
    fig.savefig(figure_paths["combined"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    _plot_trace_panel(
        display_target=display_target,
        baseline_pred=baseline_pred,
        cf_pred=cf_pred,
        ablation_cf_pred=ablation_cf_pred,
        cfg=cfg,
        trace_groups=trace_groups,
        time=time,
        slices=slices,
        save_path=figure_paths["traces"],
        y_label=y_label,
        trace_label=trace_label,
        baseline_label=baseline_label,
        cf_label=cf_label,
        ablation_label=ablation_label,
        show_ablation_traces=show_ablation_traces,
    )
    figure_paths.update(_plot_reconstruction_panel(
        categories=bar_categories,
        baseline_metrics=baseline_metrics,
        cf_metrics=cf_metrics,
        ablation_metrics=ablation_metrics,
        baseline_sem=baseline_sem,
        cf_sem=cf_sem,
        ablation_sem=ablation_sem,
        metric_key=metric_key,
        target_key=target_key,
        target_name=target_name,
        baseline_label=baseline_label,
        cf_label=cf_label,
        ablation_label=ablation_label,
        clip_ablation_bars=clip_ablation_bars,
        plot_title=plot_title,
        save_path=figure_paths["reconstruction"],
    ))
    if trajectory_arrays is not None:
        _plot_decoded_trajectory_panel(
            arrays=trajectory_arrays,
            save_path=figure_paths["trajectory"],
            display_points=display_points,
            title=plot_title.replace("reconstruction", "trajectory"),
        )
    data_paths = _export_plot_data(
        data_export_prefix=data_export_prefix,
        display_target=display_target,
        baseline_pred=baseline_pred,
        cf_pred=cf_pred,
        ablation_cf_pred=ablation_cf_pred,
        cfg=cfg,
        trace_groups=trace_groups,
        time=time,
        slices=slices,
        trace_label=trace_label,
        baseline_label=baseline_label,
        cf_label=cf_label,
        ablation_label=ablation_label,
        trajectory_arrays=trajectory_arrays,
        display_points=display_points,
    )
    if data_paths:
        figure_paths["data"] = data_paths
    return figure_paths


def _baseline_hparams(args, baseline_model):
    if baseline_model != "pivae":
        raise ValueError(f"Unsupported RatInABox baseline model: {baseline_model}")
    return {
        "seed": args.seed,
        "latent_dim": args.pivae_latent_dim,
        "gen_nodes": args.pivae_gen_nodes,
        "flow_blocks": args.pivae_flow_blocks,
        "lr": args.pivae_lr,
        "epochs": args.pivae_epochs,
        "patience": args.pivae_patience,
        "batch_size": args.pivae_batch_size,
        "samples": args.pivae_samples,
    }


def _resolve_baseline_model(args, target_key):
    if target_key != "spikes":
        raise ValueError("pi-VAE is only supported for spike-target configs.")
    return args.baseline_model


def _validate_ablation_config(main_cfg, ablation_cfg):
    fields = [
        "dataset",
        "reconstruction_target_key",
        "num_neurons",
        "num_cells",
        "include_noise_cell",
        "input_window",
        "output_window",
        "unknown_input_key",
    ]
    mismatches = []
    for field in fields:
        if getattr(main_cfg, field, None) != getattr(ablation_cfg, field, None):
            mismatches.append(
                f"{field}: main={getattr(main_cfg, field, None)!r}, ablation={getattr(ablation_cfg, field, None)!r}"
            )
    if mismatches:
        raise ValueError("Ablation config is not directly comparable:\n" + "\n".join(mismatches))


def _mean_sem(metric_rows):
    keys = metric_rows[0].keys()
    mean = {}
    sem = {}
    for key in keys:
        values = np.array([row[key] for row in metric_rows], dtype=np.float64)
        mean[key] = float(values.mean())
        sem[key] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def _metric_diff(left, right):
    return {name: float(left[name] - right[name]) for name in left}


def _load_main_model_for_log(cfg, model_log_dir):
    ckpt_dir = Path(model_log_dir) / "checkpoints"
    if not list(ckpt_dir.glob("*.ckpt")):
        raise FileNotFoundError(
            f"No checkpoint found under {ckpt_dir}. Train this fold first with scripts/train.py."
        )
    return load_model_from_ckpt(cfg, model_log_dir)


def _fit_and_predict_baseline(
    train_x_norm,
    train_y,
    val_x_norm,
    val_y,
    shift_x_norm,
    args,
    checkpoint_path,
):
    model, training = fit_pivae_baseline(
        train_u=train_x_norm,
        train_x=train_y,
        val_u=val_x_norm,
        val_x=val_y,
        args=args,
        checkpoint_path=checkpoint_path,
    )
    val_pred = predict_pivae(
        model,
        val_x_norm,
        x_dim=train_y.shape[1],
        batch_size=args.pivae_batch_size,
        n_samples=args.pivae_samples,
    )
    shift_pred = predict_pivae(
        model,
        shift_x_norm,
        x_dim=train_y.shape[1],
        batch_size=args.pivae_batch_size,
        n_samples=args.pivae_samples,
    )
    return val_pred, shift_pred, training


def evaluate_fold(cfg, args, log_dir, device, baseline_model, fold_idx=None, ablation_cfg=None, ablation_log_dir=None):
    seed_offset = 0 if fold_idx is None else fold_idx
    set_seed(args.seed + seed_offset)

    if fold_idx is None:
        model_log_dir = log_dir
        checkpoint_name = "behaviour_only_pivae.weights.h5"
    else:
        model_log_dir = f"{log_dir}/fold_{fold_idx}"
        checkpoint_name = f"behaviour_only_pivae_fold_{fold_idx}.weights.h5"

    main_model = _load_main_model_for_log(cfg, model_log_dir)
    datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx or 0)

    if fold_idx is None:
        train_dataset = datamodule.dataset("train")
        val_dataset = datamodule.dataset("val")
    else:
        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset

    target_key = getattr(cfg, "reconstruction_target_key", "rates")
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=False)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    counterfactual_batches = make_counterfactual_batches(
        datamodule,
        args.batch_size,
        args.counterfactual_mode,
    )
    source_loader = make_dict_loader(counterfactual_batches["source"], args.batch_size)
    target_loader = make_dict_loader(counterfactual_batches["target"], args.batch_size)

    train_x, train_y = collate_behaviour_and_target(train_loader, target_key)
    val_x, val_y = collate_behaviour_and_target(val_loader, target_key)
    target_x, target_y = collate_batch_behaviour_and_target(counterfactual_batches["target"], target_key)
    _, source_y = collate_batch_behaviour_and_target(counterfactual_batches["source"], target_key)
    val_display_rates = collate_display_rates(val_loader).detach().cpu().numpy() if target_key == "spikes" else None
    source_display_rates = (
        collate_batch_display_rates(counterfactual_batches["source"]).detach().cpu().numpy()
        if target_key == "spikes"
        else None
    )
    target_display_rates = (
        collate_batch_display_rates(counterfactual_batches["target"]).detach().cpu().numpy()
        if target_key == "spikes"
        else None
    )

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x_norm = (train_x - mean) / std
    val_x_norm = (val_x - mean) / std
    target_x_norm = (target_x - mean) / std

    checkpoint_path = Path(get_result_filepath(model_log_dir, checkpoint_name))
    val_pred, target_pred, training = _fit_and_predict_baseline(
        train_x_norm=train_x_norm,
        train_y=train_y,
        val_x_norm=val_x_norm,
        val_y=val_y,
        shift_x_norm=target_x_norm,
        args=args,
        checkpoint_path=checkpoint_path,
    )

    main_val_pred, main_val_true = predict_main_model(main_model, val_loader, target_key)
    cf_target_pred, cf_target_true = build_counterfactual_prediction(
        main_model, cfg, source_loader, target_loader, target_key
    )

    val_true = val_y.detach().cpu().numpy()
    source_true = source_y.detach().cpu().numpy()
    target_true = target_y.detach().cpu().numpy()
    ablation_cf_target_pred = None
    ablation_model_log_dir = None
    if ablation_cfg is not None:
        if ablation_log_dir is None:
            raise ValueError("ablation_log_dir must be provided when ablation_cfg is set.")
        ablation_model_log_dir = ablation_log_dir if fold_idx is None else f"{ablation_log_dir}/fold_{fold_idx}"
        ablation_model = _load_main_model_for_log(ablation_cfg, ablation_model_log_dir)
        ablation_cf_target_pred, ablation_cf_target_true = build_counterfactual_prediction(
            ablation_model, ablation_cfg, source_loader, target_loader, target_key
        )
        if not np.allclose(target_true, ablation_cf_target_true):
            raise ValueError("Ablated counterfactual target activity does not match the main target activity.")

    if not np.allclose(target_true, cf_target_true):
        raise ValueError("pi-VAE and counterfactual target activity do not match.")
    if not np.allclose(val_true, main_val_true):
        raise ValueError("pi-VAE and main-model validation target activity do not match.")

    if target_key == "spikes":
        baseline_val_metrics = compute_group_r2(val_display_rates, val_pred, cfg)
        main_val_metrics = compute_group_r2(val_display_rates, main_val_pred, cfg)
        baseline_target_metrics = compute_group_r2(target_display_rates, target_pred, cfg)
        cf_target_metrics = compute_group_r2(target_display_rates, cf_target_pred, cfg)
        cf_source_metrics = compute_group_r2(source_display_rates, cf_target_pred, cfg)
        ablation_cf_target_metrics = (
            compute_group_r2(target_display_rates, ablation_cf_target_pred, cfg)
            if ablation_cf_target_pred is not None
            else None
        )
        ablation_cf_source_metrics = (
            compute_group_r2(source_display_rates, ablation_cf_target_pred, cfg)
            if ablation_cf_target_pred is not None
            else None
        )
        baseline_target_bps = compute_group_bps(target_true, target_pred, cfg)
        cf_target_bps = compute_group_bps(cf_target_true, cf_target_pred, cfg)
        ablation_cf_target_bps = (
            compute_group_bps(cf_target_true, ablation_cf_target_pred, cfg)
            if ablation_cf_target_pred is not None
            else None
        )
    else:
        baseline_val_metrics = compute_group_r2(val_true, val_pred, cfg)
        main_val_metrics = compute_group_r2(main_val_true, main_val_pred, cfg)
        baseline_target_metrics = compute_group_r2(target_true, target_pred, cfg)
        cf_target_metrics = compute_group_r2(cf_target_true, cf_target_pred, cfg)
        cf_source_metrics = compute_group_r2(source_true, cf_target_pred, cfg)
        ablation_cf_target_metrics = (
            compute_group_r2(cf_target_true, ablation_cf_target_pred, cfg)
            if ablation_cf_target_pred is not None
            else None
        )
        ablation_cf_source_metrics = (
            compute_group_r2(source_true, ablation_cf_target_pred, cfg)
            if ablation_cf_target_pred is not None
            else None
        )
        baseline_target_bps = None
        cf_target_bps = None
        ablation_cf_target_bps = None

    metrics = {
        "fold_idx": fold_idx,
        "model_log_dir": model_log_dir,
        "ablation_model_log_dir": ablation_model_log_dir,
        "counterfactual_mode": args.counterfactual_mode,
        "source_dataset_flag": counterfactual_batches["mode"]["source_flag"],
        "target_dataset_flag": counterfactual_batches["mode"]["target_flag"],
        "target_reversed": counterfactual_batches["mode"]["target_reverse"],
        "baseline_checkpoint": str(checkpoint_path),
        "training": training,
        "baseline_val_r2": baseline_val_metrics,
        "main_val_r2": main_val_metrics,
        "main_minus_baseline_val_r2": _metric_diff(main_val_metrics, baseline_val_metrics),
        "baseline_shift_target_rates_r2": baseline_target_metrics,
        "counterfactual_shift_target_rates_r2": cf_target_metrics,
        "counterfactual_shift_source_rates_r2": cf_source_metrics,
        "counterfactual_minus_baseline_shift_target_rates_r2": _metric_diff(
            cf_target_metrics, baseline_target_metrics
        ),
        "baseline_counterfactual_target_rates_r2": baseline_target_metrics,
        "counterfactual_target_rates_r2": cf_target_metrics,
        "counterfactual_source_rates_r2": cf_source_metrics,
        "counterfactual_minus_baseline_target_rates_r2": _metric_diff(
            cf_target_metrics, baseline_target_metrics
        ),
    }
    if ablation_cf_target_metrics is not None:
        metrics["ablation_counterfactual_shift_target_rates_r2"] = ablation_cf_target_metrics
        metrics["ablation_counterfactual_shift_source_rates_r2"] = ablation_cf_source_metrics
        metrics["ablation_counterfactual_minus_baseline_shift_target_rates_r2"] = _metric_diff(
            ablation_cf_target_metrics, baseline_target_metrics
        )
        metrics["ablation_counterfactual_target_rates_r2"] = ablation_cf_target_metrics
        metrics["ablation_counterfactual_source_rates_r2"] = ablation_cf_source_metrics
        metrics["ablation_counterfactual_minus_baseline_target_rates_r2"] = _metric_diff(
            ablation_cf_target_metrics, baseline_target_metrics
        )
    if target_key == "spikes":
        metrics["baseline_shift_target_bps"] = baseline_target_bps
        metrics["counterfactual_shift_target_bps"] = cf_target_bps
        metrics["counterfactual_minus_baseline_shift_target_bps"] = _metric_diff(
            cf_target_bps, baseline_target_bps
        )
        metrics["baseline_counterfactual_target_bps"] = baseline_target_bps
        metrics["counterfactual_target_bps"] = cf_target_bps
        metrics["counterfactual_minus_baseline_target_bps"] = _metric_diff(
            cf_target_bps, baseline_target_bps
        )
        if ablation_cf_target_bps is not None:
            metrics["ablation_counterfactual_shift_target_bps"] = ablation_cf_target_bps
            metrics["ablation_counterfactual_minus_baseline_shift_target_bps"] = _metric_diff(
                ablation_cf_target_bps, baseline_target_bps
            )
            metrics["ablation_counterfactual_target_bps"] = ablation_cf_target_bps
            metrics["ablation_counterfactual_minus_baseline_target_bps"] = _metric_diff(
                ablation_cf_target_bps, baseline_target_bps
            )
    else:
        metrics["baseline_shift_target_r2"] = baseline_target_metrics
        metrics["counterfactual_shift_target_r2"] = cf_target_metrics
        metrics["counterfactual_minus_baseline_shift_target_r2"] = _metric_diff(
            cf_target_metrics, baseline_target_metrics
        )
        metrics["baseline_counterfactual_target_r2"] = baseline_target_metrics
        metrics["counterfactual_target_r2"] = cf_target_metrics
        metrics["counterfactual_minus_baseline_target_r2"] = _metric_diff(
            cf_target_metrics, baseline_target_metrics
        )
        if ablation_cf_target_metrics is not None:
            metrics["ablation_counterfactual_shift_target_r2"] = ablation_cf_target_metrics
            metrics["ablation_counterfactual_minus_baseline_shift_target_r2"] = _metric_diff(
                ablation_cf_target_metrics, baseline_target_metrics
            )
            metrics["ablation_counterfactual_target_r2"] = ablation_cf_target_metrics
            metrics["ablation_counterfactual_minus_baseline_target_r2"] = _metric_diff(
                ablation_cf_target_metrics, baseline_target_metrics
            )

    return {
        "metrics": metrics,
        "target_activity": target_true,
        "display_rates": target_display_rates,
        "baseline_pred": target_pred,
        "cf_pred": cf_target_pred,
        "ablation_cf_pred": ablation_cf_target_pred,
        "source_position": counterfactual_batches["source"]["pos"],
        "target_position": counterfactual_batches["target"]["pos"],
    }


def aggregate_fold_metrics(fold_metrics):
    metric_groups = [
        "baseline_val_r2",
        "main_val_r2",
        "main_minus_baseline_val_r2",
        "baseline_shift_target_rates_r2",
        "counterfactual_shift_target_rates_r2",
        "counterfactual_shift_source_rates_r2",
        "counterfactual_minus_baseline_shift_target_rates_r2",
        "ablation_counterfactual_shift_target_rates_r2",
        "ablation_counterfactual_shift_source_rates_r2",
        "ablation_counterfactual_minus_baseline_shift_target_rates_r2",
        "counterfactual_target_rates_r2",
        "counterfactual_source_rates_r2",
        "ablation_counterfactual_target_rates_r2",
        "ablation_counterfactual_source_rates_r2",
        "baseline_shift_target_bps",
        "counterfactual_shift_target_bps",
        "counterfactual_minus_baseline_shift_target_bps",
        "ablation_counterfactual_shift_target_bps",
        "ablation_counterfactual_minus_baseline_shift_target_bps",
    ]
    aggregate = {}
    for group in metric_groups:
        rows = [fold[group] for fold in fold_metrics if group in fold and fold[group] is not None]
        if not rows:
            continue
        mean, sem = _mean_sem(rows)
        aggregate[group] = {"mean": mean, "sem": sem}
    return aggregate


FIXED_ARGS = {
    "baseline_model": "pivae",
    "batch_size": 512,
    "seed": 42,
    "display_points": 1500,
    "pivae_latent_dim": 4,
    "pivae_gen_nodes": 60,
    "pivae_flow_blocks": 2,
    "pivae_lr": 5e-4,
    "pivae_epochs": 600,
    "pivae_patience": 40,
    "pivae_batch_size": 256,
    "pivae_samples": 32,
    "cebra_model_architecture": "offset10-model",
    "cebra_batch_size": 1000,
    "cebra_learning_rate": 1e-4,
    "cebra_temperature": 1.0,
    "cebra_output_dimension": 8,
    "cebra_max_iterations": 15000,
    "cebra_distance": "cosine",
    "cebra_conditional": "time_delta",
    "cebra_time_offsets": 10,
    "cebra_knn_neighbors": 36,
    "cebra_knn_metric": "cosine",
    "cebra_verbose": False,
    "trajectory_decode_activity": "auto",
    "skip_decoded_trajectory": False,
    "behaviour_preservation": False,
    "cf_label": "Neural-DISCO",
    "ablation_label": "w/o reg.",
    "cv": True,
    "cv_folds": 10,
    "show_ablation_traces": False,
    "plot_groups": None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=DEFAULT_MAIN_CONFIG)
    parser.add_argument(
        "--ablation-config",
        default=None,
        help="Optional ablated model config to add as a w/o reg. comparison.",
    )
    parser.add_argument(
        "--counterfactual-mode",
        choices=sorted(COUNTERFACTUAL_MODES),
        default="position_shift",
        help="RatInABox counterfactual dataset pair to evaluate.",
    )
    parser.add_argument("--force-baseline-retrain", action="store_true")
    args = parser.parse_args()
    for key, value in FIXED_ARGS.items():
        setattr(args, key, value)

    set_seed(args.seed)
    cfg = load_config(args.config)
    if args.cv_folds is not None:
        cfg.cv_folds = args.cv_folds
    if args.behaviour_preservation:
        args.plot_groups = ["speed"]
    target_key = getattr(cfg, "reconstruction_target_key", "rates")
    log_dir = get_log_dir(cfg, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_info = COUNTERFACTUAL_MODES[args.counterfactual_mode]
    plot_title = "Behaviour Preservation" if args.behaviour_preservation else mode_info["title"]
    suptitle = plot_title if args.behaviour_preservation else mode_info["suptitle"]

    baseline_model = _resolve_baseline_model(args, target_key)
    baseline_label = "pi-VAE"
    ablation_cfg = None
    ablation_log_dir = None
    if args.ablation_config is not None:
        ablation_cfg = load_config(args.ablation_config)
        if args.cv_folds is not None:
            ablation_cfg.cv_folds = args.cv_folds
        _validate_ablation_config(cfg, ablation_cfg)
        ablation_log_dir = get_log_dir(ablation_cfg, args.ablation_config)
    figure_stem = _figure_stem(target_key, args.counterfactual_mode, args.behaviour_preservation)
    if args.cv:
        num_folds = int(getattr(cfg, "cv_folds", 5))
        if ablation_cfg is not None and int(getattr(ablation_cfg, "cv_folds", 1)) < num_folds:
            raise ValueError(
                f"Ablation config has fewer CV folds ({getattr(ablation_cfg, 'cv_folds', 1)}) than main config ({num_folds})."
            )
        fold_results = [
            evaluate_fold(
                cfg=cfg,
                args=args,
                log_dir=log_dir,
                device=device,
                baseline_model=baseline_model,
                fold_idx=fold_idx,
                ablation_cfg=ablation_cfg,
                ablation_log_dir=ablation_log_dir,
            )
            for fold_idx in range(num_folds)
        ]
        fold_metrics = [result["metrics"] for result in fold_results]
        aggregate = aggregate_fold_metrics(fold_metrics)
        display_result = fold_results[0]
        trajectory_arrays = None
        if not args.skip_decoded_trajectory and not args.behaviour_preservation:
            trajectory_arrays = build_decoded_position_trajectory(
                cfg=cfg,
                args=args,
                log_dir=log_dir,
                device=device,
                fold_idx=0,
            )
        metric_overrides = {
            "metric_key": "r2",
            "baseline_mean": aggregate["baseline_shift_target_rates_r2"]["mean"],
            "baseline_sem": aggregate["baseline_shift_target_rates_r2"]["sem"],
            "counterfactual_mean": aggregate["counterfactual_shift_target_rates_r2"]["mean"],
            "counterfactual_sem": aggregate["counterfactual_shift_target_rates_r2"]["sem"],
        }
        if "ablation_counterfactual_shift_target_rates_r2" in aggregate:
            metric_overrides["ablation_counterfactual_mean"] = aggregate[
                "ablation_counterfactual_shift_target_rates_r2"
            ]["mean"]
            metric_overrides["ablation_counterfactual_sem"] = aggregate[
                "ablation_counterfactual_shift_target_rates_r2"
            ]["sem"]
        figure_stem = f"{figure_stem}_cv"
        if ablation_cfg is not None:
            figure_stem = f"{figure_stem}_pivae_ablation" if baseline_model == "pivae" else f"{figure_stem}_ablation"
        result_path = get_result_filepath(log_dir, f"{figure_stem}_metrics.json")
        figure_path = get_visualization_filepath(log_dir, f"{figure_stem}.png")

        metrics = {
            "config": str(args.config),
            "reconstruction_target_key": target_key,
            "device": str(device),
            "counterfactual_mode": args.counterfactual_mode,
            "figure_title": plot_title,
            "baseline_type": f"{baseline_model}_behaviour_only",
            "ablation_config": args.ablation_config,
            "ablation_log_dir": ablation_log_dir,
            "cv": True,
            "cv_folds": num_folds,
            "baseline_hparams": _baseline_hparams(args, baseline_model),
            "trajectory_decode_activity": args.trajectory_decode_activity,
            "fold_metrics": fold_metrics,
            "aggregate": aggregate,
            "figure_path": figure_path,
        }
        figure_paths = plot_comparison(
            target_activity=display_result["target_activity"],
            baseline_pred=display_result["baseline_pred"],
            cf_pred=display_result["cf_pred"],
            cfg=cfg,
            save_path=figure_path,
            display_points=args.display_points,
            target_key=target_key,
            display_rates=display_result["display_rates"],
            metric_overrides=metric_overrides,
            ablation_cf_pred=display_result["ablation_cf_pred"],
            baseline_label=baseline_label,
            cf_label=args.cf_label,
            ablation_label=args.ablation_label,
            show_ablation_traces=args.show_ablation_traces,
            plot_groups=args.plot_groups,
            plot_title=plot_title,
            suptitle=suptitle,
            trajectory_arrays=trajectory_arrays,
            data_export_prefix=Path(result_path).with_name(
                f"{Path(result_path).stem.replace('_metrics', '')}_plot_data"
            ),
        )
        metrics["figure_paths"] = figure_paths
    else:
        result = evaluate_fold(
            cfg=cfg,
            args=args,
            log_dir=log_dir,
            device=device,
            baseline_model=baseline_model,
            fold_idx=None,
            ablation_cfg=ablation_cfg,
            ablation_log_dir=ablation_log_dir,
        )
        if ablation_cfg is not None:
            figure_stem = f"{figure_stem}_pivae_ablation" if baseline_model == "pivae" else f"{figure_stem}_ablation"
        result_path = get_result_filepath(log_dir, f"{figure_stem}_metrics.json")
        figure_path = get_visualization_filepath(log_dir, f"{figure_stem}.png")
        trajectory_arrays = None
        if not args.skip_decoded_trajectory and not args.behaviour_preservation:
            trajectory_arrays = build_decoded_position_trajectory(
                cfg=cfg,
                args=args,
                log_dir=log_dir,
                device=device,
                fold_idx=None,
            )
        metrics = {
            "config": str(args.config),
            "reconstruction_target_key": target_key,
            "device": str(device),
            "counterfactual_mode": args.counterfactual_mode,
            "figure_title": plot_title,
            "baseline_type": f"{baseline_model}_behaviour_only",
            "ablation_config": args.ablation_config,
            "ablation_log_dir": ablation_log_dir,
            "cv": False,
            "baseline_hparams": _baseline_hparams(args, baseline_model),
            "trajectory_decode_activity": args.trajectory_decode_activity,
            **result["metrics"],
            "figure_path": figure_path,
        }
        figure_paths = plot_comparison(
            target_activity=result["target_activity"],
            baseline_pred=result["baseline_pred"],
            cf_pred=result["cf_pred"],
            cfg=cfg,
            save_path=figure_path,
            display_points=args.display_points,
            target_key=target_key,
            display_rates=result["display_rates"],
            ablation_cf_pred=result["ablation_cf_pred"],
            baseline_label=baseline_label,
            cf_label=args.cf_label,
            ablation_label=args.ablation_label,
            show_ablation_traces=args.show_ablation_traces,
            plot_groups=args.plot_groups,
            plot_title=plot_title,
            suptitle=suptitle,
            trajectory_arrays=trajectory_arrays,
            data_export_prefix=Path(result_path).with_name(
                f"{Path(result_path).stem.replace('_metrics', '')}_plot_data"
            ),
        )
        metrics["figure_paths"] = figure_paths

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
