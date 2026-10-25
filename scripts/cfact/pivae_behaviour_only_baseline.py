import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score

# MC Maze pi-VAE baseline only. Allen pi-VAE runs live in
# allen_speed_mean_baselines.py, which imports the collate and decode-feature
# helpers below.
current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.cfact.mc_maze_mlp_decode_xy import (  # noqa: E402
    _ci95,
    _normalize_between_controls,
    build_relaxed_pairs,
    build_trial_windows,
    decoder_predict,
    fit_decoder,
    generated_activity_to_decoder_features,
    prepare_behaviour_for_model_input,
)
from scripts.cfact.mc_maze_pairing import PAIRING_STRATEGY  # noqa: E402
from scripts.cfact.rat_mlp_behaviour_only_baseline import (  # noqa: E402
    bits_per_spike,
    fit_pivae_baseline,
    predict_pivae,
    set_seed,
)
from src.data.collate import collate_all_batches  # noqa: E402
from src.eval.mc_maze_windows import window_condition_target  # noqa: E402
from src.plotting import COUNTERFACTUAL_COLORS  # noqa: E402
from src.utils import (  # noqa: E402
    get_log_dir,
    get_result_filepath,
    get_visualization_filepath,
    load_config,
    load_datamodule_from_config,
)


def _loader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


def _flatten_spikes(spikes):
    if torch.is_tensor(spikes):
        spikes = spikes.detach().cpu().numpy()
    spikes = np.asarray(spikes, dtype=np.float32)
    if spikes.ndim == 3 and spikes.shape[-1] == 1:
        return spikes.squeeze(-1)
    if spikes.ndim == 3 and spikes.shape[1] == 1:
        return spikes.squeeze(1)
    if spikes.ndim == 2:
        return spikes
    raise ValueError(f"Unsupported spike shape {spikes.shape}")


def _collate_behaviour_u(batch, behaviours):
    values = []
    for behaviour in behaviours:
        value = batch[behaviour].float()
        values.append(value.reshape(value.shape[0], -1))
    return torch.cat(values, dim=1)


def _collate_dataset_for_pivae(dataset, behaviours, batch_size):
    batch = collate_all_batches(_loader(dataset, batch_size))
    u = _collate_behaviour_u(batch, behaviours)
    x = torch.from_numpy(_flatten_spikes(batch["spikes"])).float()
    valid = torch.isfinite(u).all(dim=1) & torch.isfinite(x).all(dim=1)
    return u[valid], x[valid]


def _pooled_r2(rows, true_key, pred_key):
    true = np.concatenate([row[true_key] for row in rows], axis=0)
    pred = np.concatenate([row[pred_key] for row in rows], axis=0)
    return float(r2_score(true, pred))


def _bootstrap_pair_metrics(pair_arrays, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    metrics = {
        "source_to_target_r2": _pooled_r2(pair_arrays, "target_true", "source_decoded"),
        "target_to_target_r2": _pooled_r2(pair_arrays, "target_true", "target_decoded"),
        "pivae_to_source_r2": _pooled_r2(pair_arrays, "source_true", "pivae_decoded"),
        "pivae_to_target_r2": _pooled_r2(pair_arrays, "target_true", "pivae_decoded"),
    }
    metrics["pivae_gain_to_target"] = metrics["pivae_to_target_r2"] - metrics["source_to_target_r2"]
    metrics["pivae_gap_closed"] = _normalize_between_controls(
        metrics["pivae_to_target_r2"],
        metrics["source_to_target_r2"],
        metrics["target_to_target_r2"],
    )
    if n_bootstrap <= 1 or len(pair_arrays) <= 1:
        for key in list(metrics):
            metrics[f"{key}_ci95"] = 0.0
        return metrics

    boot = {key: [] for key in metrics}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(pair_arrays), size=len(pair_arrays))
        rows = [pair_arrays[i] for i in idx]
        source_to_target = _pooled_r2(rows, "target_true", "source_decoded")
        target_to_target = _pooled_r2(rows, "target_true", "target_decoded")
        pivae_to_source = _pooled_r2(rows, "source_true", "pivae_decoded")
        pivae_to_target = _pooled_r2(rows, "target_true", "pivae_decoded")
        boot["source_to_target_r2"].append(source_to_target)
        boot["target_to_target_r2"].append(target_to_target)
        boot["pivae_to_source_r2"].append(pivae_to_source)
        boot["pivae_to_target_r2"].append(pivae_to_target)
        boot["pivae_gain_to_target"].append(pivae_to_target - source_to_target)
        boot["pivae_gap_closed"].append(
            _normalize_between_controls(pivae_to_target, source_to_target, target_to_target)
        )
    for key, vals in boot.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        metrics[f"{key}_ci95"] = _ci95(vals) if len(vals) else float("nan")
    return metrics


def _mc_train_decoder_inputs(train_dataset, behaviours, cfg):
    spikes = train_dataset.spikes.data.numpy()
    spikes = _flatten_spikes(spikes)
    behaviour_mode = (
        cfg.behaviour_target_mode
        if "behaviour_target_mode" in cfg
        else (cfg.spike_target_mode if "spike_target_mode" in cfg else "future")
    )
    behaviour_targets = {
        behaviour: window_condition_target(
            train_dataset.data_loader.load_behaviour(behaviour),
            input_window=cfg.input_window,
            output_window=cfg.output_window,
            delay_factor=0,
            target_mode=behaviour_mode,
        )
        for behaviour in behaviours
    }
    return spikes, behaviour_targets


def _mc_target_u(dataset, behaviours, target_behaviour):
    values = [
        prepare_behaviour_for_model_input(dataset, behaviour, target_behaviour[behaviour])
        for behaviour in behaviours
    ]
    return np.concatenate(values, axis=1).astype(np.float32)


def _mc_fit_decoders(cfg, args, datamodule):
    train_spikes, train_behaviour = _mc_train_decoder_inputs(datamodule.dataset("train"), cfg.behaviours, cfg)
    return {
        behaviour: fit_decoder(
            train_spikes,
            train_behaviour[behaviour],
            decoder_type=args.decoder,
            hidden_dim=args.decoder_hidden_dim,
            decoder_kwargs=_decoder_kwargs(args),
        )
        for behaviour in cfg.behaviours
    }


def _allen_rates_to_decoder_features(rates, args):
    if args.allen_pivae_decode_features == "rates":
        return rates
    if args.allen_pivae_decode_features == "smooth_rates":
        return gaussian_filter1d(rates, sigma=args.allen_spike_smoothing, axis=0)
    if args.allen_pivae_decode_features == "poisson_smooth":
        sampled = torch.poisson(torch.from_numpy(rates).clamp_min(0.0)).detach().cpu().numpy()
        return gaussian_filter1d(sampled, sigma=args.allen_spike_smoothing, axis=0)
    raise ValueError(f"Unsupported Allen pi-VAE decode feature mode: {args.allen_pivae_decode_features}")


def _mc_fold(cfg, args, fold_idx, log_dir, decoders=None):
    set_seed(args.seed + fold_idx)
    datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx)
    fold_train_dataset = datamodule.train_dataset if args.cv else datamodule.dataset("train")
    fold_val_dataset = datamodule.val_dataset if args.cv else datamodule.train_dataset
    eval_dataset = datamodule.dataset("test")

    train_u, train_x = _collate_dataset_for_pivae(fold_train_dataset, cfg.behaviours, args.batch_size)
    val_u, val_x = _collate_dataset_for_pivae(fold_val_dataset, cfg.behaviours, args.batch_size)
    ckpt_name = f"pivae_behaviour_only_fold_{fold_idx}.weights.h5" if args.cv else "pivae_behaviour_only.weights.h5"
    checkpoint_path = Path(get_result_filepath(f"{log_dir}/fold_{fold_idx}" if args.cv else log_dir, ckpt_name))
    pivae_model, training = fit_pivae_baseline(
        train_u=train_u,
        train_x=train_x,
        val_u=val_u,
        val_x=val_x,
        args=args,
        checkpoint_path=checkpoint_path,
    )

    if decoders is None:
        decoders = _mc_fit_decoders(cfg, args, datamodule)

    trial_info = eval_dataset.data_loader.load_trial_info().reset_index(drop=True)
    pairs = build_relaxed_pairs(trial_info)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("No MC Maze relaxed pairs available for pi-VAE evaluation.")

    pair_records = []
    target_u_all = []
    lengths = []
    for source_idx, target_idx in pairs:
        source_inputs, source_behaviour = build_trial_windows(
            eval_dataset,
            source_idx,
            cfg.behaviours,
            cfg.input_window,
            cfg.output_window,
            unknown_input_window=(cfg.unknown_input_window if "unknown_input_window" in cfg else None),
            spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
            behaviour_target_mode=(cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None),
        )
        target_inputs, target_behaviour = build_trial_windows(
            eval_dataset,
            target_idx,
            cfg.behaviours,
            cfg.input_window,
            cfg.output_window,
            unknown_input_window=(cfg.unknown_input_window if "unknown_input_window" in cfg else None),
            spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
            behaviour_target_mode=(cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None),
        )
        target_u = _mc_target_u(eval_dataset, cfg.behaviours, target_behaviour)
        pair_records.append(
            {
                "source_inputs": source_inputs,
                "target_inputs": target_inputs,
                "source_behaviour": source_behaviour,
                "target_behaviour": target_behaviour,
            }
        )
        target_u_all.append(target_u)
        lengths.append(len(target_u))

    pivae_rates_cat = predict_pivae(
        pivae_model,
        np.concatenate(target_u_all, axis=0),
        x_dim=train_x.shape[1],
        batch_size=args.pivae_batch_size,
        n_samples=args.pivae_samples,
    )
    offsets = np.cumsum([0] + lengths)

    per_pair_arrays = {behaviour: [] for behaviour in cfg.behaviours}
    target_spikes_all = []
    pivae_rates_all = []
    for idx, record in enumerate(pair_records):
        pivae_rates = pivae_rates_cat[offsets[idx] : offsets[idx + 1]]
        pivae_decode_features = generated_activity_to_decoder_features(
            pivae_rates,
            mode=args.mc_generated_decode_features,
            seed=args.seed + fold_idx * 1000003 + idx,
        )
        source_spikes = _flatten_spikes(record["source_inputs"]["spikes"])
        target_spikes = _flatten_spikes(record["target_inputs"]["spikes"])
        target_spikes_all.append(target_spikes)
        pivae_rates_all.append(pivae_rates)
        for behaviour in cfg.behaviours:
            decoder = decoders[behaviour]
            per_pair_arrays[behaviour].append(
                {
                    "source_true": record["source_behaviour"][behaviour],
                    "target_true": record["target_behaviour"][behaviour],
                    "source_decoded": decoder_predict(decoder, source_spikes),
                    "target_decoded": decoder_predict(decoder, target_spikes),
                    "pivae_decoded": decoder_predict(decoder, pivae_decode_features),
                }
            )

    summary = {
        behaviour: _bootstrap_pair_metrics(
            per_pair_arrays[behaviour],
            n_bootstrap=args.summary_bootstrap_samples,
            seed=args.seed + fold_idx,
        )
        for behaviour in cfg.behaviours
    }
    target_spikes = np.concatenate(target_spikes_all, axis=0)
    pivae_rates = np.concatenate(pivae_rates_all, axis=0)
    summary["spikes"] = {
        "pivae_bps_to_target_spikes": bits_per_spike(pivae_rates, target_spikes),
    }
    return {
        "fold_idx": fold_idx,
        "baseline_checkpoint": str(checkpoint_path),
        "training": training,
        "num_pairs": len(pairs),
        "pairing_strategy": PAIRING_STRATEGY,
        "summary": summary,
    }


def _aggregate_cv(fold_results):
    fold_summaries = [result["summary"] for result in fold_results]
    keys = fold_summaries[0].keys()
    aggregate = {}
    for group in keys:
        aggregate[group] = {}
        metric_keys = fold_summaries[0][group].keys()
        for metric in metric_keys:
            values = np.asarray([fold[group][metric] for fold in fold_summaries], dtype=np.float64)
            aggregate[group][metric] = {
                "mean": float(values.mean()),
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
    return aggregate


def _decoder_kwargs(args):
    return {
        "num_layers": args.decoder_num_layers,
        "activation": args.decoder_activation,
        "dropout": args.decoder_dropout,
        "lr": args.decoder_lr,
        "weight_decay": args.decoder_weight_decay,
        "max_epochs": args.decoder_max_epochs,
        "patience": args.decoder_patience,
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
        "cebra_verbose": args.cebra_verbose,
    }


def _plot_summary(cfg, metrics, out_path, cv):
    behaviours = list(cfg.behaviours)
    labels = [b.replace("_", " ").title() for b in behaviours]
    values = [metrics[b]["pivae_to_target_r2"] for b in behaviours]
    yerr = [metrics[b].get("pivae_to_target_r2_ci95", 0.0) for b in behaviours]
    if cv and isinstance(values[0], dict):
        yerr = [metrics[b]["pivae_to_target_r2"]["sem"] for b in behaviours]
        values = [metrics[b]["pivae_to_target_r2"]["mean"] for b in behaviours]
    ylabel = r"Decode $R^2$ to target"

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=yerr, color=COUNTERFACTUAL_COLORS["counterfactual"], capsize=3)
    ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title("pi-VAE behaviour-only baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


FIXED_ARGS = {
    "cv": True,
    "cv_folds": 10,
    "max_pairs": None,
    "batch_size": 512,
    "seed": 42,
    "summary_bootstrap_samples": 500,
    "decoder": "auto",
    "decoder_hidden_dim": 256,
    "decoder_num_layers": 3,
    "decoder_activation": "gelu",
    "decoder_dropout": 0.0,
    "decoder_lr": 1e-3,
    "decoder_weight_decay": 1e-4,
    "decoder_max_epochs": 300,
    "decoder_patience": 20,
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
    "pivae_latent_dim": 4,
    "pivae_gen_nodes": 60,
    "pivae_flow_blocks": 2,
    "pivae_lr": 5e-4,
    "pivae_epochs": 600,
    "pivae_patience": 40,
    "pivae_batch_size": 256,
    "pivae_samples": 32,
    "mc_generated_decode_features": "poisson",
    "figure_name": None,
    "result_name": None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--force-baseline-retrain", action="store_true")
    args = parser.parse_args()
    for key, value in FIXED_ARGS.items():
        setattr(args, key, value)

    set_seed(args.seed)
    cfg = load_config(args.config)
    if args.cv_folds is not None:
        cfg.cv_folds = args.cv_folds
    if cfg.dataset != "mc_maze":
        raise ValueError(
            f"pi-VAE behaviour-only baseline supports MC Maze only, got dataset {cfg.dataset!r}. "
            "Run Allen pi-VAE through scripts/cfact/allen_speed_mean_baselines.py."
        )
    if args.decoder == "auto":
        args.decoder = "cebra"

    log_dir = get_log_dir(cfg, args.config)
    num_folds = int(getattr(cfg, "cv_folds", 5)) if args.cv else 1
    fold_results = []
    mc_decoders = None
    if args.cv:
        decoder_datamodule = load_datamodule_from_config(cfg, fold_idx=0)
        mc_decoders = _mc_fit_decoders(cfg, args, decoder_datamodule)
    for fold_idx in range(num_folds):
        fold_results.append(_mc_fold(cfg, args, fold_idx, log_dir, decoders=mc_decoders))

    aggregate = _aggregate_cv(fold_results) if args.cv else fold_results[0]["summary"]
    stem = "pivae_behaviour_only_baseline_cv" if args.cv else "pivae_behaviour_only_baseline"
    result_path = get_result_filepath(log_dir, args.result_name or f"{stem}_metrics.json")
    figure_path = get_visualization_filepath(log_dir, args.figure_name or f"{stem}.png")

    result = {
        "config": args.config,
        "dataset": cfg.dataset,
        "cv": bool(args.cv),
        "cv_folds": num_folds if args.cv else None,
        "decoder": args.decoder,
        "mc_generated_decode_features": args.mc_generated_decode_features,
        "mc_pairing_strategy": PAIRING_STRATEGY,
        "allen_pivae_decode_features": None,
        "baseline_type": "pivae_behaviour_only",
        "pivae_hparams": {
            "latent_dim": args.pivae_latent_dim,
            "gen_nodes": args.pivae_gen_nodes,
            "flow_blocks": args.pivae_flow_blocks,
            "lr": args.pivae_lr,
            "epochs": args.pivae_epochs,
            "patience": args.pivae_patience,
            "batch_size": args.pivae_batch_size,
            "samples": args.pivae_samples,
        },
        "fold_results": fold_results,
        "aggregate": aggregate,
        "figure_path": figure_path,
    }
    _plot_summary(cfg, aggregate, figure_path, cv=args.cv)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    compact = {
        "config": result["config"],
        "dataset": result["dataset"],
        "cv": result["cv"],
        "cv_folds": result["cv_folds"],
        "decoder": result["decoder"],
        "baseline_type": result["baseline_type"],
        "aggregate": result["aggregate"],
        "figure_path": figure_path,
        "result_path": result_path,
    }
    print(json.dumps(compact, indent=2))
    print(f"Saved figure to {figure_path}")
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
