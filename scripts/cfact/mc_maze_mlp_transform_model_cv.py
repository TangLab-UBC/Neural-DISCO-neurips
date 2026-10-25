import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.cfact.mc_maze_mlp_decode_xy import (  # noqa: E402
    CEBRAEmbeddingDecoder,
    compute_pair_metrics,
    evaluate_reconstruction,
    fit_shared_decoders,
)
from scripts.cfact.mc_maze_mlp_transform import (  # noqa: E402
    LOCKED_DISPLAY_FAMILY,
    build_families,
    choose_display_subset,
    choose_secondary_display_subset,
    choose_trajectory_grid_rows,
    plot_figure,
    plot_summary_figure,
    plot_trajectory_grid,
    summarize_families,
    summarize_transform_cv,
)
from scripts.cfact.mc_maze_pairing import PAIRING_STRATEGY  # noqa: E402
from src.utils import (  # noqa: E402
    get_log_dir,
    get_result_filepath,
    get_visualization_filepath,
    load_config,
    load_datamodule_from_config,
    load_module_from_config,
)


DEFAULT_CONFIG = "configs/mc_maze/mlp_kinematics_eye_pos_iw1_tuned_cv10.yaml"


def _load_fold_module(cfg, fold_log_dir, device):
    ckpt_dir = Path(fold_log_dir) / "checkpoints"
    ckpt_files = glob.glob(str(ckpt_dir / "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No trained fold checkpoint found under {ckpt_dir}")
    ckpt_path = max(ckpt_files, key=os.path.getmtime)
    module = load_module_from_config(cfg).to(device)
    import munch

    torch.serialization.add_safe_globals([munch.Munch])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    module.load_state_dict(ckpt["state_dict"])
    module.eval()
    module.requires_grad_(False)
    return module, ckpt_path


def _mean_sem(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(arr))
    finite = arr[np.isfinite(arr)]
    sem = float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
    return mean, sem


def _aggregate_fold_summaries(fold_summaries, behaviours):
    aggregate = {}
    for behaviour in behaviours:
        rows = [summary[behaviour] for summary in fold_summaries]
        metric_keys = sorted(
            key
            for key, value in rows[0].items()
            if isinstance(value, (int, float)) and not key.endswith("_ci95")
        )
        out = {
            "summary_method": "model_cv_over_folds",
            "cv_folds": len(fold_summaries),
        }
        for key in metric_keys:
            mean, sem = _mean_sem([row.get(key, np.nan) for row in rows])
            out[key] = mean
            out[f"{key}_sem"] = sem
            if key.endswith("_r2"):
                out[key.replace("_r2", "_ci95")] = 1.96 * sem
            elif key in {"target_gain_to_target", "cf_gain_to_target", "cf_gap_closed"}:
                out[f"{key}_ci95"] = 1.96 * sem
        aggregate[behaviour] = out
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
        "seed": 0,
        "summary_bootstrap_samples": args.summary_bootstrap_samples,
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


def _figure_name_with_suffix(figure_name, suffix):
    path = Path(figure_name)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix or '.png'}"))


def _decoder_cache_metadata(args, fold_idx, cfg, decoder_kwargs, train_dataset):
    return {
        "dataset": str(cfg.dataset),
        "data_filepath": str(cfg.data_filepath),
        "fold_idx": int(fold_idx),
        "cv_folds": int(cfg.cv_folds),
        "train_spikes_shape": list(train_dataset.spikes.data.shape),
        "behaviours": list(cfg.behaviours),
        "input_window": int(cfg.input_window),
        "output_window": int(cfg.output_window),
        "spike_target_mode": cfg.spike_target_mode if "spike_target_mode" in cfg else None,
        "behaviour_target_mode": cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None,
        "decoder": args.decoder,
        "hidden_dim": int(args.hidden_dim),
        "decoder_kwargs": decoder_kwargs,
    }


def _decoder_cache_path(metadata):
    payload = json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return get_result_filepath("logs/mc_maze_decoder_cache", f"shared_decoders_{digest}")


def _load_cached_cebra_model(path):
    """Load a trusted, locally generated CEBRA decoder cache.

    CEBRA 0.1.0 delegates to ``torch.load(path)``. PyTorch 2.6 changed the
    default to ``weights_only=True``, which cannot deserialize CEBRA's saved
    estimator object. These cache files are written by this script, so load
    them explicitly as trusted objects and retain CEBRA's type check.
    """
    import cebra

    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, cebra.CEBRA):
        raise RuntimeError(f"Cached model is not a CEBRA estimator: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.solver_.to(device)
    model.device = device
    model.device_ = device
    return model


def _load_cached_decoders(cache_path, metadata):
    cache_dir = Path(cache_path)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        cached_metadata = json.load(f)
    if cached_metadata != metadata:
        print(f"Ignoring stale decoder cache with mismatched metadata: {cache_path}")
        return None

    if metadata["decoder"] == "cebra":
        decoders = {}
        for behaviour in metadata["behaviours"]:
            decoder = CEBRAEmbeddingDecoder()
            decoder.model = _load_cached_cebra_model(cache_dir / f"{behaviour}_cebra.pt")
            decoder.decoder = torch.load(
                cache_dir / f"{behaviour}_knn.pt",
                map_location="cpu",
                weights_only=False,
            )["decoder"]
            decoders[behaviour] = decoder
        return decoders

    payload_path = cache_dir / "decoders.pt"
    if not payload_path.exists():
        return None
    return torch.load(payload_path, map_location="cpu", weights_only=False)["decoders"]


def _save_cached_decoders(cache_path, metadata, decoders):
    cache_dir = Path(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if metadata["decoder"] == "cebra":
        for behaviour, decoder in decoders.items():
            decoder.model.save(str(cache_dir / f"{behaviour}_cebra.pt"))
            torch.save({"decoder": decoder.decoder}, cache_dir / f"{behaviour}_knn.pt")
        return

    torch.save({"decoders": decoders}, cache_dir / "decoders.pt")


def _load_or_fit_decoders(args, fold_idx, cfg, train_dataset, decoder_kwargs):
    metadata = _decoder_cache_metadata(args, fold_idx, cfg, decoder_kwargs, train_dataset)
    cache_path = _decoder_cache_path(metadata)
    if not args.disable_decoder_cache and Path(cache_path).exists():
        decoders = _load_cached_decoders(cache_path, metadata)
        if decoders is not None:
            print(f"Loaded cached {args.decoder} decoders from {cache_path}")
            return decoders, cache_path, True

    print(f"Fitting shared {args.decoder} decoders for fold {fold_idx}; cache={cache_path}")
    decoders = fit_shared_decoders(
        train_dataset,
        cfg.behaviours,
        cfg.input_window,
        cfg.output_window,
        decoder_type=args.decoder,
        hidden_dim=args.hidden_dim,
        decoder_kwargs=decoder_kwargs,
        spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
        behaviour_target_mode=(cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None),
    )
    if not args.disable_decoder_cache:
        _save_cached_decoders(cache_path, metadata, decoders)
        print(f"Saved shared {args.decoder} decoders to {cache_path}")
    return decoders, cache_path, False


def _compute_pair_metrics(
    args,
    fold_idx,
    module,
    train_dataset,
    test_dataset,
    cfg,
    device,
    decoder_kwargs,
    decoders=None,
):
    if decoders is None:
        decoders, _, _ = _load_or_fit_decoders(args, fold_idx, cfg, train_dataset, decoder_kwargs)

    per_pair, fitted_decoders = compute_pair_metrics(
        module,
        train_dataset,
        test_dataset,
        cfg.behaviours,
        device,
        cfg.input_window,
        cfg.output_window,
        unknown_input_window=(cfg.unknown_input_window if "unknown_input_window" in cfg else None),
        spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
        behaviour_target_mode=(cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None),
        decoder_type=args.decoder,
        hidden_dim=args.hidden_dim,
        decoder_kwargs=decoder_kwargs,
        decoders=decoders,
        generated_decode_features=args.generated_decode_features,
        generated_seed=fold_idx * 10000019,
    )
    return per_pair, fitted_decoders


def _render_figures(
    cfg,
    log_dir,
    figure_name,
    trajectory_figure_name,
    summary_figure_name,
    per_pair,
    display_groups,
    display_meta,
    context_meta,
    aggregate_summary,
    panel_mode,
    pivae_summary=None,
    neds_summary=None,
    ablation_summary=None,
    trajectory_examples=4,
    trajectory_selection_mode="deterministic",
    trajectory_selection_seed=0,
):
    fig_path = get_visualization_filepath(log_dir, figure_name)
    display_bounds = plot_figure(
        cfg,
        display_groups,
        display_meta,
        context_meta,
        aggregate_summary,
        fig_path,
        panel_mode=panel_mode,
        pivae_summary=pivae_summary,
        neds_summary=neds_summary,
        ablation_summary=ablation_summary,
    )

    display_behaviour = "hand_pos" if "hand_pos" in cfg.behaviours else cfg.behaviours[0]
    curated_rows = choose_trajectory_grid_rows(
        per_pair,
        display_behaviour,
        n_examples=trajectory_examples,
        selection_mode=trajectory_selection_mode,
        seed=trajectory_selection_seed,
    )
    trajectory_background_rows = per_pair
    trajectory_name = trajectory_figure_name or _figure_name_with_suffix(figure_name, "trajectories")
    summary_name = summary_figure_name or _figure_name_with_suffix(figure_name, "decode_r2")
    trajectory_path = get_visualization_filepath(log_dir, trajectory_name)
    summary_path = get_visualization_filepath(log_dir, summary_name)
    trajectory_bounds = plot_trajectory_grid(
        cfg,
        curated_rows,
        trajectory_path,
        panel_mode=panel_mode,
        background_rows=trajectory_background_rows,
        selection_mode=trajectory_selection_mode,
        selection_seed=trajectory_selection_seed,
    )
    summary_bounds = plot_summary_figure(
        cfg,
        aggregate_summary,
        summary_path,
        pivae_summary=pivae_summary,
        neds_summary=neds_summary,
        ablation_summary=ablation_summary,
    )
    return {
        "combined": fig_path,
        "trajectories": trajectory_path,
        "decode_r2": summary_path,
        "display_bounds": display_bounds,
        "trajectory_bounds": trajectory_bounds,
        "summary_bounds": summary_bounds,
    }


def _choose_display_groups(per_pair, cfg):
    display_behaviour = "hand_pos" if "hand_pos" in cfg.behaviours else cfg.behaviours[0]
    families = build_families(per_pair, display_behaviour, n_bins=8)
    family_summary = summarize_families(families, display_behaviour)
    family_rows, display_meta, ranked_display_candidates = choose_display_subset(
        families,
        display_behaviour,
        min_family_pairs=10,
        min_context_pairs=4,
        locked_display=LOCKED_DISPLAY_FAMILY,
    )
    display_groups = [family_rows]
    secondary_display_meta = []
    secondary_rows, secondary_meta = choose_secondary_display_subset(
        families,
        display_behaviour,
        display_meta,
        min_position_pairs=4,
    )
    if secondary_rows is not None:
        display_groups.append(secondary_rows)
        secondary_display_meta.append(dict(secondary_meta))

    display_meta = dict(display_meta)
    display_meta["num_bins"] = 8
    display_meta["secondary_display"] = secondary_display_meta
    context_meta = {
        "used_context_grouping": bool(display_meta.get("used_context_grouping", False)),
        "context": display_meta.get("context"),
        "context_group_count": int(display_meta.get("context_group_count", 0)),
        "used_position_grouping": bool(display_meta.get("used_position_grouping", False)),
        "position_pair": display_meta.get("position_pair"),
        "position_group_count": int(display_meta.get("position_group_count", 0)),
    }
    return display_groups, display_meta, context_meta, family_summary, ranked_display_candidates


FIXED_ARGS = {
    "decoder": "cebra",
    "hidden_dim": 256,
    "decoder_num_layers": 3,
    "decoder_activation": "gelu",
    "decoder_dropout": 0.0,
    "decoder_lr": 1e-3,
    "decoder_weight_decay": 1e-4,
    "decoder_max_epochs": 300,
    "decoder_patience": 20,
    "summary_bootstrap_samples": 0,
    "generated_decode_features": "poisson",
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
    "top_panel_mode": "all_plus_mean",
    "figure_name": "counterfactual_transform_model_cv.png",
    "result_name": "counterfactual_transform_model_cv_metrics.json",
    "trajectory_figure_name": None,
    "summary_figure_name": None,
    "trajectory_examples": 4,
    "trajectory_selection_mode": "deterministic",
    "trajectory_selection_seed": 0,
    "disable_decoder_cache": False,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fold-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of CV folds to evaluate. Defaults to all folds.",
    )
    parser.add_argument(
        "--pivae-metrics",
        default=None,
        help="Optional pi-VAE metrics JSON to add as a behaviour-only baseline bar.",
    )
    parser.add_argument(
        "--neds-metrics",
        default=None,
        help="Optional NEDS metrics JSON to add as a behaviour-only baseline bar.",
    )
    parser.add_argument(
        "--ablation-metrics",
        default=None,
        help="Optional no-regularization ablation metrics JSON to add as a w/o reg. bar.",
    )
    args = parser.parse_args()
    for key, value in FIXED_ARGS.items():
        setattr(args, key, value)

    cfg = load_config(args.config)
    if "cv_folds" not in cfg:
        raise ValueError("Model-CV evaluation requires a config with cv_folds.")
    if args.fold_indices is None:
        fold_indices = list(range(int(cfg.cv_folds)))
    else:
        fold_indices = [int(fold_idx) for fold_idx in args.fold_indices]
        for fold_idx in fold_indices:
            if fold_idx < 0 or fold_idx >= int(cfg.cv_folds):
                raise ValueError(f"fold-indices must be in [0, {int(cfg.cv_folds) - 1}], got {fold_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_dir = get_log_dir(cfg, args.config)
    decoder_kwargs = _decoder_kwargs(args)
    pivae_summary = None
    if args.pivae_metrics is not None:
        with open(args.pivae_metrics, "r", encoding="utf-8") as f:
            pivae_metrics = json.load(f)
        pivae_mode = pivae_metrics.get("mc_generated_decode_features")
        if pivae_mode is not None and pivae_mode != args.generated_decode_features:
            raise ValueError(
                "pi-VAE metrics use mc_generated_decode_features="
                f"{pivae_mode!r}, but Neural-DISCO eval uses {args.generated_decode_features!r}."
            )
        pivae_summary = pivae_metrics["aggregate"]
    neds_summary = None
    if args.neds_metrics is not None:
        with open(args.neds_metrics, "r", encoding="utf-8") as f:
            neds_metrics = json.load(f)
        neds_mode = neds_metrics.get("mc_generated_decode_features")
        if neds_mode is not None and neds_mode != args.generated_decode_features:
            raise ValueError(
                "NEDS metrics use mc_generated_decode_features="
                f"{neds_mode!r}, but Neural-DISCO eval uses {args.generated_decode_features!r}."
            )
        neds_summary = neds_metrics["aggregate"]
    ablation_summary = None
    if args.ablation_metrics is not None:
        with open(args.ablation_metrics, "r", encoding="utf-8") as f:
            ablation_metrics = json.load(f)
        ablation_mode = ablation_metrics.get("generated_decode_features")
        if ablation_mode is not None and ablation_mode != args.generated_decode_features:
            raise ValueError(
                "Ablation metrics use generated_decode_features="
                f"{ablation_mode!r}, but Neural-DISCO eval uses {args.generated_decode_features!r}."
            )
        ablation_summary = ablation_metrics.get("ablation_summary") or ablation_metrics.get("summary_metrics")
        if ablation_summary is None:
            raise ValueError(
                f"No ablation_summary or summary_metrics found in {args.ablation_metrics}"
            )

    fold_summaries = []
    fold_results = []
    first_per_pair = None
    first_display_groups = None
    first_reconstruction = None
    first_display_meta = None
    first_context_meta = None
    first_family_summary = None
    first_ranked_candidates = None

    for fold_idx in fold_indices:
        print(f"Evaluating MC Maze CV fold {fold_idx + 1}/{int(cfg.cv_folds)}")
        fold_log_dir = f"{log_dir}/fold_{fold_idx}"
        module, ckpt_path = _load_fold_module(cfg, fold_log_dir, device)
        datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx)
        train_dataset = datamodule.dataset("train")
        test_dataset = datamodule.dataset("test")
        reconstruction_metrics = evaluate_reconstruction(module, test_dataset, device)
        per_pair, _ = _compute_pair_metrics(
            args,
            fold_idx,
            module,
            train_dataset,
            test_dataset,
            cfg,
            device,
            decoder_kwargs,
        )
        summary = summarize_transform_cv(
            train_dataset,
            per_pair,
            cfg.behaviours,
            cfg.input_window,
            cfg.output_window,
            spike_target_mode=(cfg.spike_target_mode if "spike_target_mode" in cfg else None),
            behaviour_target_mode=(cfg.behaviour_target_mode if "behaviour_target_mode" in cfg else None),
            decoder_type=args.decoder,
            hidden_dim=args.hidden_dim,
            cv_folds=1,
            cv_repeats=1,
            decoder_kwargs=decoder_kwargs,
        )
        fold_summaries.append(summary)
        fold_results.append(
            {
                "fold_idx": fold_idx,
                "checkpoint": str(ckpt_path),
                "reconstruction": reconstruction_metrics,
                "summary_metrics": summary,
                "num_pairs": len(per_pair),
            }
        )

        if first_per_pair is None:
            first_per_pair = per_pair
            first_reconstruction = reconstruction_metrics
            (
                first_display_groups,
                first_display_meta,
                first_context_meta,
                first_family_summary,
                first_ranked_candidates,
            ) = _choose_display_groups(per_pair, cfg)
        print(f"Finished fold {fold_idx + 1}/{int(cfg.cv_folds)}")

    aggregate_summary = _aggregate_fold_summaries(fold_summaries, cfg.behaviours)
    figure_outputs = _render_figures(
        cfg,
        log_dir,
        args.figure_name,
        args.trajectory_figure_name,
        args.summary_figure_name,
        first_per_pair,
        first_display_groups,
        first_display_meta,
        first_context_meta,
        aggregate_summary,
        panel_mode=args.top_panel_mode,
        pivae_summary=pivae_summary,
        neds_summary=neds_summary,
        ablation_summary=ablation_summary,
        trajectory_examples=args.trajectory_examples,
        trajectory_selection_mode=args.trajectory_selection_mode,
        trajectory_selection_seed=args.trajectory_selection_seed,
    )

    result = {
        "config": args.config,
        "device": device.type,
        "decoder": args.decoder,
        "generated_decode_features": args.generated_decode_features,
        "pairing_strategy": PAIRING_STRATEGY,
        "display_pairing_strategy": PAIRING_STRATEGY,
        "trajectory_selection_mode": args.trajectory_selection_mode,
        "trajectory_selection_seed": int(args.trajectory_selection_seed),
        "summary_method": "model_cv_over_folds",
        "cv_folds": int(cfg.cv_folds),
        "evaluated_folds": fold_indices,
        "fold_results": fold_results,
        "summary_metrics": aggregate_summary,
        "pivae_metrics_path": args.pivae_metrics,
        "neds_metrics_path": args.neds_metrics,
        "ablation_metrics_path": args.ablation_metrics,
        "display_bounds": figure_outputs["display_bounds"],
        "trajectory_bounds": figure_outputs["trajectory_bounds"],
        "summary_bounds": figure_outputs["summary_bounds"],
        "display_family": first_display_meta,
        "top_ranked_families": first_family_summary[:10],
        "top_ranked_display_candidates": first_ranked_candidates,
        "representative_reconstruction": first_reconstruction,
        "figure_path": figure_outputs["combined"],
        "trajectory_figure_path": figure_outputs["trajectories"],
        "decode_r2_figure_path": figure_outputs["decode_r2"],
    }
    result_path = get_result_filepath(log_dir, args.result_name)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved model-CV figure to {figure_outputs['combined']}")
    print(f"Saved trajectory figure to {figure_outputs['trajectories']}")
    print(f"Saved decode-R2 figure to {figure_outputs['decode_r2']}")
    print(f"Saved model-CV metrics to {result_path}")


if __name__ == "__main__":
    main()
