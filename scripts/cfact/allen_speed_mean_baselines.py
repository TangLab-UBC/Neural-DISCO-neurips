import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.cfact.mc_maze_mlp_decode_xy import decoder_predict, fit_decoder  # noqa: E402
from scripts.cfact.neds_behaviour_only_baseline import (  # noqa: E402
    _build_neds_model,
    _fit_neds_input_normalizer,
    _predict_neds_from_behaviour,
    _resolve_device,
    _serializable_normalizer,
    _train_neds,
)
from scripts.cfact.pivae_behaviour_only_baseline import (  # noqa: E402
    _allen_rates_to_decoder_features,
    _collate_dataset_for_pivae,
    _flatten_spikes,
    _loader,
)
from scripts.cfact.rat_mlp_behaviour_only_baseline import (  # noqa: E402
    bits_per_spike,
    fit_pivae_baseline,
    predict_pivae,
    set_seed,
)
from src.data.collate import collate_all_batches  # noqa: E402
from src.plotting import COUNTERFACTUAL_SUMMARY_COLORS  # noqa: E402
from src.utils import (  # noqa: E402
    get_log_dir,
    get_result_filepath,
    get_visualization_filepath,
    load_config,
    load_datamodule_from_config,
    load_model_from_ckpt,
)


STIMULUS_TARGETS = ("temporal_frequency",)
EVALUATION_SEED = 161803


class AllenPresentationSequenceDataset(Dataset):
    """Group Allen bin-level samples into full drifting-grating presentations for NEDS."""

    def __init__(self, bin_dataset, behaviours, target_names=("orientation", "temporal_frequency", "condition")):
        base_dataset, base_indices = self._unwrap_subset(bin_dataset)
        if not hasattr(base_dataset, "presentation_ids") or not hasattr(base_dataset, "presentation_bins"):
            raise ValueError("Allen NEDS sequence baseline requires presentation_id/presentation_bin metadata.")

        self.base_dataset = base_dataset
        self.behaviours = list(behaviours)
        self.target_names = list(target_names)
        base_indices = np.asarray(base_indices, dtype=np.int64)
        presentation_ids = base_dataset.presentation_ids[base_indices].detach().cpu().numpy()
        presentation_bins = base_dataset.presentation_bins[base_indices].detach().cpu().numpy()

        grouped = {}
        for base_idx, presentation_id, presentation_bin in zip(base_indices, presentation_ids, presentation_bins):
            grouped.setdefault(int(presentation_id), []).append((int(presentation_bin), int(base_idx)))

        lengths = np.asarray([len(rows) for rows in grouped.values()], dtype=np.int64)
        if len(lengths) == 0:
            raise ValueError("No Allen presentations available for NEDS sequence dataset.")
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        self.seq_len = int(unique_lengths[np.argmax(counts)])
        self.groups = []
        self.presentation_ids = []
        dropped = 0
        for presentation_id in sorted(grouped):
            rows = sorted(grouped[presentation_id], key=lambda row: row[0])
            if len(rows) != self.seq_len:
                dropped += 1
                continue
            expected_bins = np.arange(self.seq_len)
            actual_bins = np.asarray([row[0] for row in rows])
            if not np.array_equal(actual_bins, expected_bins):
                dropped += 1
                continue
            self.groups.append([row[1] for row in rows])
            self.presentation_ids.append(presentation_id)
        if not self.groups:
            raise ValueError("No complete Allen presentation sequences available for NEDS.")
        if dropped:
            print(
                f"Note: dropped {dropped} incomplete Allen presentations; using "
                f"{len(self.groups)} presentations with {self.seq_len} bins each.",
                flush=True,
            )

    @staticmethod
    def _unwrap_subset(dataset):
        if isinstance(dataset, Subset):
            base_dataset, base_indices = AllenPresentationSequenceDataset._unwrap_subset(dataset.dataset)
            return base_dataset, np.asarray(base_indices, dtype=np.int64)[np.asarray(dataset.indices, dtype=np.int64)]
        return dataset, np.arange(len(dataset), dtype=np.int64)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        items = [self.base_dataset[base_idx] for base_idx in self.groups[idx]]
        out = {
            "spikes": torch.stack([item["spikes"].squeeze(-1).float() for item in items], dim=0),
            "presentation_id": torch.tensor(self.presentation_ids[idx], dtype=torch.long),
        }
        for key in self.behaviours + self.target_names:
            out[key] = torch.stack([item[key].reshape(-1).float() for item in items], dim=0)
        return out


def _decoder_kwargs(args, fold_idx=0):
    return {
        "num_layers": args.decoder_num_layers,
        "activation": args.decoder_activation,
        "dropout": args.decoder_dropout,
        "lr": args.decoder_lr,
        "weight_decay": args.decoder_weight_decay,
        "max_epochs": args.decoder_max_epochs,
        "patience": args.decoder_patience,
        "seed": args.seed + fold_idx,
    }


def _as_2d(batch, key):
    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    return value.reshape(value.shape[0], -1)


def _as_1d(batch, key):
    return _as_2d(batch, key).reshape(-1)


def _rmse(actual, predicted):
    actual = np.asarray(actual, dtype=np.float32).reshape(len(actual), -1)
    predicted = np.asarray(predicted, dtype=np.float32).reshape(len(predicted), -1)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _pack(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "values": values.astype(float).tolist(),
    }


def _fit_stimulus_classifier(train_features, train_labels):
    valid = np.isfinite(train_labels)
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels[valid])
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(train_features[valid], y_train)
    return clf, le


def _score_stimulus_classifier(clf, le, features, labels):
    valid = np.isfinite(labels)
    y_true = le.transform(labels[valid])
    y_pred = clf.predict(features[valid])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_classes": int(len(le.classes_)),
    }


def _fit_stimulus_decoders(train_spikes, train_batch):
    decoders = {}
    for target in STIMULUS_TARGETS:
        labels = _as_1d(train_batch, target)
        decoders[target] = _fit_stimulus_classifier(train_spikes, labels)
    return decoders


def _score_stimulus_decoders(decoders, features, test_batch):
    out = {}
    for target, (clf, le) in decoders.items():
        out[target] = _score_stimulus_classifier(clf, le, features, _as_1d(test_batch, target))
    return out


def _baseline_feature_args(args):
    return SimpleNamespace(
        allen_pivae_decode_features=args.generated_decode_features,
        allen_spike_smoothing=args.allen_spike_smoothing,
    )


def _speed_mean(train_batch):
    return float(np.nanmean(_as_1d(train_batch, "speed")))


def _fold_log_dir(log_dir, args, fold_idx):
    return f"{log_dir}/fold_{fold_idx}" if args.cv else log_dir


def _pivae_checkpoint_path(log_dir, args, fold_idx):
    ckpt_name = (
        f"pivae_speed_mean_behaviour_only_fold_{fold_idx}.weights.h5"
        if args.cv
        else "pivae_speed_mean_behaviour_only.weights.h5"
    )
    return Path(get_result_filepath(_fold_log_dir(log_dir, args, fold_idx), ckpt_name))


def _neds_checkpoint_path(log_dir, args, fold_idx):
    norm_tag = "_norm"
    seq_tag = f"_seq{int(args.neds_max_F)}" if int(args.neds_max_F or 1) > 1 else ""
    arch_tag = "_nedsdefault"
    readout_tag = "_registeredout"
    mode_tag = "_encoding_encoding"
    if args.cv:
        name = f"neds{mode_tag}{norm_tag}{seq_tag}{arch_tag}{readout_tag}_fold_{fold_idx}.pt"
    else:
        name = f"neds{mode_tag}{norm_tag}{seq_tag}{arch_tag}{readout_tag}.pt"
    return Path(get_result_filepath(_fold_log_dir(log_dir, args, fold_idx), name))


def _counterfactual_behaviour_arrays(cfg, test_batch, speed_mean):
    arrays = {
        behaviour: _as_2d(test_batch, behaviour)
        for behaviour in cfg.behaviours
    }
    if "speed" not in arrays:
        raise ValueError("Allen speed-to-mean baseline requires 'speed' in cfg.behaviours.")
    arrays["speed"] = np.full_like(arrays["speed"], speed_mean, dtype=np.float32)
    return arrays


def _counterfactual_sequence_arrays(cfg, test_sequence_batch, speed_mean):
    arrays = {
        behaviour: test_sequence_batch[behaviour].detach().cpu().numpy().astype(np.float32)
        for behaviour in cfg.behaviours
    }
    if "speed" not in arrays:
        raise ValueError("Allen speed-to-mean baseline requires 'speed' in cfg.behaviours.")
    arrays["speed"] = np.full_like(arrays["speed"], speed_mean, dtype=np.float32)
    return arrays


def _sequence_loader(dataset, batch_size, shuffle=False, num_workers=0):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _make_neds_sequence_datasets(cfg, args, datamodule):
    train_bin_dataset = datamodule.train_dataset if args.cv else datamodule.dataset("train")
    val_bin_dataset = datamodule.val_dataset if args.cv else datamodule.train_dataset
    train_sequence_dataset = AllenPresentationSequenceDataset(train_bin_dataset, cfg.behaviours)
    val_sequence_dataset = AllenPresentationSequenceDataset(val_bin_dataset, cfg.behaviours)
    test_sequence_dataset = AllenPresentationSequenceDataset(datamodule.dataset("test"), cfg.behaviours)
    seq_lengths = {
        train_sequence_dataset.seq_len,
        val_sequence_dataset.seq_len,
        test_sequence_dataset.seq_len,
    }
    if len(seq_lengths) != 1:
        raise ValueError(f"Allen NEDS sequence length mismatch across splits: {sorted(seq_lengths)}")
    args.neds_max_F = train_sequence_dataset.seq_len
    return train_sequence_dataset, val_sequence_dataset, test_sequence_dataset


def _evaluate_generated_rates(
    generated_rates,
    train_batch,
    test_batch,
    train_spikes,
    test_spikes,
    speed_target,
    args,
    fold_idx,
):
    feature_args = _baseline_feature_args(args)
    torch.manual_seed(EVALUATION_SEED + fold_idx)
    generated_features = _allen_rates_to_decoder_features(generated_rates, feature_args)
    # Decoder initialization; its validation split uses args.seed + fold_idx.
    torch.manual_seed(EVALUATION_SEED + fold_idx)
    speed_decoder = fit_decoder(
        train_spikes,
        _as_2d(train_batch, "speed"),
        decoder_type=args.speed_decoder,
        hidden_dim=args.decoder_hidden_dim,
        decoder_kwargs=_decoder_kwargs(args, fold_idx),
    )
    source_speed_decoded = np.asarray(decoder_predict(speed_decoder, test_spikes)).reshape(len(test_spikes), -1)
    generated_speed_decoded = np.asarray(decoder_predict(speed_decoder, generated_features)).reshape(
        len(test_spikes), -1
    )
    stimulus_decoders = _fit_stimulus_decoders(train_spikes, train_batch)
    return {
        "speed_rmse": {
            "source_to_source": _rmse(_as_2d(test_batch, "speed"), source_speed_decoded),
            "source_to_mean_target": _rmse(speed_target, source_speed_decoded),
            "generated_to_mean_target": _rmse(speed_target, generated_speed_decoded),
        },
        "stimulus_decoding": {
            "source_spikes": _score_stimulus_decoders(stimulus_decoders, test_spikes, test_batch),
            "generated": _score_stimulus_decoders(stimulus_decoders, generated_features, test_batch),
        },
        "spikes": {
            "bps_to_test_spikes": bits_per_spike(
                generated_rates,
                _flatten_spikes(test_batch["spikes"]),
            )
        },
    }


def _run_pivae_fold(cfg, args, fold_idx, log_dir, datamodule, train_batch, test_batch, train_spikes, test_spikes):
    fold_train_dataset = datamodule.train_dataset if args.cv else datamodule.dataset("train")
    fold_val_dataset = datamodule.val_dataset if args.cv else datamodule.train_dataset
    train_u, train_x = _collate_dataset_for_pivae(fold_train_dataset, cfg.behaviours, args.batch_size)
    val_u, val_x = _collate_dataset_for_pivae(fold_val_dataset, cfg.behaviours, args.batch_size)
    checkpoint_path = _pivae_checkpoint_path(log_dir, args, fold_idx)
    set_seed(args.seed + fold_idx)
    model, training = fit_pivae_baseline(
        train_u=train_u,
        train_x=train_x,
        val_u=val_u,
        val_x=val_x,
        args=args,
        checkpoint_path=checkpoint_path,
    )
    speed_mean = _speed_mean(train_batch)
    cf_arrays = _counterfactual_behaviour_arrays(cfg, test_batch, speed_mean)
    cf_u = np.concatenate([cf_arrays[name] for name in cfg.behaviours], axis=1).astype(np.float32)
    np.random.seed(args.seed + fold_idx)
    rates = predict_pivae(
        model,
        cf_u,
        x_dim=train_x.shape[1],
        batch_size=args.pivae_batch_size,
        n_samples=args.pivae_samples,
    )
    speed_target = np.full((len(test_spikes), 1), speed_mean, dtype=np.float32)
    metrics = _evaluate_generated_rates(
        rates,
        train_batch,
        test_batch,
        train_spikes,
        test_spikes,
        speed_target,
        args,
        fold_idx,
    )
    return {
        "fold_idx": int(fold_idx),
        "baseline": "pivae",
        "checkpoint_path": str(checkpoint_path),
        "training": training,
        "speed_mean": speed_mean,
        "summary": metrics,
    }


def _run_neds_fold(cfg, args, fold_idx, log_dir, datamodule, train_batch, test_batch, train_spikes, test_spikes):
    test_dataset = datamodule.dataset("test")
    num_neurons = int(test_dataset[0]["spikes"].shape[0])
    train_sequence_dataset, val_sequence_dataset, test_sequence_dataset = _make_neds_sequence_datasets(
        cfg, args, datamodule
    )
    device = _resolve_device(args)
    checkpoint_path = _neds_checkpoint_path(log_dir, args, fold_idx)
    args.neds_static_behaviour_encoder = True
    normalizer = _fit_neds_input_normalizer(train_sequence_dataset, cfg.behaviours, args.loader_batch_size)
    set_seed(args.seed + fold_idx)
    model, specs, eid = _build_neds_model(args, cfg, num_neurons, device)
    train_loader = _sequence_loader(
        train_sequence_dataset,
        args.neds_sequence_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = _sequence_loader(
        val_sequence_dataset,
        args.neds_sequence_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    if checkpoint_path.exists() and not args.force_baseline_retrain:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        training = {
            "checkpoint_path": str(checkpoint_path),
            "best_epoch": int(state.get("epoch", -1)),
            "best_val_loss": float(state.get("val_loss", float("nan"))),
            "early_stop_patience": int(state.get("early_stop_patience", args.neds_patience)),
            "early_stop_min_delta": float(state.get("early_stop_min_delta", args.neds_min_delta)),
            "history": state.get("history", []),
            "loaded_from_cache": True,
        }
    else:
        training = _train_neds(model, train_loader, val_loader, specs, device, eid, args, log_dir, fold_idx, normalizer)
    speed_mean = _speed_mean(train_batch)
    test_sequence_batch = collate_all_batches(_loader(test_sequence_dataset, args.loader_batch_size))
    cf_arrays = _counterfactual_sequence_arrays(cfg, test_sequence_batch, speed_mean)
    sequence_rates = _predict_neds_from_behaviour(
        model,
        cf_arrays,
        specs,
        device,
        eid,
        num_neurons,
        args.neds_predict_batch_size,
        normalizer,
    )
    rates = sequence_rates.reshape(-1, sequence_rates.shape[-1])
    if len(rates) != len(test_spikes):
        raise ValueError(
            f"NEDS sequence output has {len(rates)} bins but test split has {len(test_spikes)} bins."
        )
    speed_target = np.full((len(test_spikes), 1), speed_mean, dtype=np.float32)
    metrics = _evaluate_generated_rates(
        rates,
        train_batch,
        test_batch,
        train_spikes,
        test_spikes,
        speed_target,
        args,
        fold_idx,
    )
    return {
        "fold_idx": int(fold_idx),
        "baseline": "neds",
        "checkpoint_path": training["checkpoint_path"],
        "training": training,
        "input_normalizer": _serializable_normalizer(normalizer),
        "n_presentations": len(test_sequence_dataset),
        "presentation_bins": int(test_sequence_dataset.seq_len),
        "speed_mean": speed_mean,
        "summary": metrics,
    }


def _run_neural_disco_fold(
    model_cfg,
    model_config_path,
    baseline_name,
    args,
    fold_idx,
    train_batch,
    test_batch,
    train_spikes,
    test_spikes,
):
    model_log_dir = get_log_dir(model_cfg, model_config_path)
    model_fold_dir = f"{model_log_dir}/fold_{fold_idx}" if args.cv else model_log_dir
    model = load_model_from_ckpt(model_cfg, model_fold_dir)
    speed_mean = _speed_mean(train_batch)
    cf_batch = {
        "spikes": test_batch["spikes"],
    }
    for behaviour in model_cfg.behaviours:
        cf_batch[behaviour] = test_batch[behaviour]
    cf_batch["speed"] = torch.ones_like(test_batch["speed"]) * speed_mean
    with torch.no_grad():
        rates = model(cf_batch)["reconstruction"].squeeze().detach().cpu().numpy()
    speed_target = np.full((len(test_spikes), 1), speed_mean, dtype=np.float32)
    metrics = _evaluate_generated_rates(
        rates,
        train_batch,
        test_batch,
        train_spikes,
        test_spikes,
        speed_target,
        args,
        fold_idx,
    )
    return {
        "fold_idx": int(fold_idx),
        "baseline": baseline_name,
        "model_config": str(model_config_path),
        "model_log_dir": model_fold_dir,
        "speed_mean": speed_mean,
        "summary": metrics,
    }


def _run_gaussian_glm_fold(
    model_cfg,
    model_config_path,
    args,
    fold_idx,
    glm_train_batch,
    glm_test_batch,
    eval_train_batch,
    eval_test_batch,
    train_spikes,
    test_spikes,
):
    """Evaluate the source-residual Gaussian GLM edit.

    The GLM config standardizes its behaviour inputs separately from the
    evaluation data. The edit is therefore constructed in GLM
    coordinates, while the RMSE target must remain in the physical speed units
    used to train the decoder.
    """
    model_log_dir = get_log_dir(model_cfg, model_config_path)
    model_fold_dir = f"{model_log_dir}/fold_{fold_idx}" if args.cv else model_log_dir
    model = load_model_from_ckpt(model_cfg, model_fold_dir)

    edit_speed_mean = _speed_mean(glm_train_batch)
    physical_speed_mean = _speed_mean(eval_train_batch)
    cf_batch = {
        "spikes": glm_test_batch["spikes"],
    }
    for behaviour in model_cfg.behaviours:
        cf_batch[behaviour] = glm_test_batch[behaviour]
    cf_batch["speed"] = (
        torch.ones_like(glm_test_batch["speed"]) * edit_speed_mean
    )

    with torch.no_grad():
        source_spikes = (
            glm_test_batch["spikes"].squeeze().detach().cpu().numpy()
        )
        glm_source = (
            model(glm_test_batch)["reconstruction"]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        glm_target = (
            model(cf_batch)["reconstruction"]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        rates = np.clip(source_spikes - glm_source + glm_target, 0.0, None)

    speed_target = np.full(
        (len(test_spikes), 1),
        physical_speed_mean,
        dtype=np.float32,
    )
    metrics = _evaluate_generated_rates(
        rates,
        eval_train_batch,
        eval_test_batch,
        train_spikes,
        test_spikes,
        speed_target,
        args,
        fold_idx,
    )
    return {
        "fold_idx": int(fold_idx),
        "baseline": "gaussian_glm",
        "implementation": "source_conditioned_gaussian_glm",
        "model_config": str(model_config_path),
        "model_log_dir": model_fold_dir,
        "edit_speed_mean_normalized": edit_speed_mean,
        "evaluation_speed_mean_physical": physical_speed_mean,
        "summary": metrics,
    }


def _aggregate(fold_results):
    out = {}
    for baseline in sorted({row["baseline"] for row in fold_results}):
        rows = [row for row in fold_results if row["baseline"] == baseline]
        out[baseline] = {
            "speed_rmse": {
                key: _pack([row["summary"]["speed_rmse"][key] for row in rows])
                for key in rows[0]["summary"]["speed_rmse"]
            },
            "spikes": {
                key: _pack([row["summary"]["spikes"][key] for row in rows])
                for key in rows[0]["summary"]["spikes"]
            },
            "stimulus_decoding": {},
        }
        for source in ("source_spikes", "generated"):
            out[baseline]["stimulus_decoding"][source] = {}
            for target in STIMULUS_TARGETS:
                out[baseline]["stimulus_decoding"][source][target] = {
                    metric: _pack(
                        [row["summary"]["stimulus_decoding"][source][target][metric] for row in rows]
                    )
                    for metric in ("accuracy", "balanced_accuracy")
                }
                out[baseline]["stimulus_decoding"][source][target]["n_classes"] = int(
                    rows[0]["summary"]["stimulus_decoding"][source][target]["n_classes"]
                )
    return out


def _plot_summary(aggregate, out_path):
    baselines = [
        b
        for b in ("neural_disco", "no_reg", "gaussian_glm", "pivae", "neds")
        if b in aggregate
    ]
    labels = {
        "neural_disco": "Neural-DISCO",
        "no_reg": "w/o reg.",
        "gaussian_glm": "Gaussian GLM",
        "pivae": "pi-VAE",
        "neds": "NEDS",
    }
    colors = {
        "neural_disco": COUNTERFACTUAL_SUMMARY_COLORS.get("main_effect", "#f58518"),
        "no_reg": COUNTERFACTUAL_SUMMARY_COLORS.get("ablation", "#9a9a9a"),
        "gaussian_glm": "#4c78a8",
        "pivae": COUNTERFACTUAL_SUMMARY_COLORS.get("baseline", "#2ca02c"),
        "neds": COUNTERFACTUAL_SUMMARY_COLORS.get("neds_baseline", "#cc79a7"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), gridspec_kw={"width_ratios": [1.15, 0.85]})

    x = np.arange(len(baselines))
    speed_vals = [aggregate[b]["speed_rmse"]["generated_to_mean_target"]["mean"] for b in baselines]
    speed_err = [aggregate[b]["speed_rmse"]["generated_to_mean_target"]["sem"] for b in baselines]
    axes[0].bar(x, speed_vals, yerr=speed_err, color=[colors[b] for b in baselines], capsize=3)
    source_to_mean = float(
        np.mean([aggregate[b]["speed_rmse"]["source_to_mean_target"]["mean"] for b in baselines])
    )
    axes[0].axhline(source_to_mean, color="#222222", linestyle=":", linewidth=2.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[b] for b in baselines])
    axes[0].set_ylabel("Speed decode RMSE to mean")
    axes[0].set_title("Speed removal")

    stim_label_lookup = {
        "orientation": "Orientation",
        "temporal_frequency": "Temporal frequency",
    }
    stim_targets = [
        target
        for target in STIMULUS_TARGETS
        if target in aggregate[baselines[0]]["stimulus_decoding"]["generated"]
    ]
    stim_labels = [stim_label_lookup.get(target, target) for target in stim_targets]
    xpos = np.arange(len(stim_labels))
    group_width = 0.62
    width = group_width / max(len(baselines), 1)
    source_vals = [
        np.mean([aggregate[b]["stimulus_decoding"]["source_spikes"][target]["balanced_accuracy"]["mean"] for b in baselines])
        for target in stim_targets
    ]
    axes[1].hlines(source_vals, xpos - group_width / 2, xpos + group_width / 2, colors="#222222", linestyles=":", linewidth=2.5)
    chance_vals = [1.0 / aggregate[baselines[0]]["stimulus_decoding"]["generated"][target]["n_classes"] for target in stim_targets]
    axes[1].hlines(chance_vals, xpos - group_width / 2, xpos + group_width / 2, colors="#777777", linestyles="--", linewidth=1.4)
    for idx, baseline in enumerate(baselines):
        vals = [
            aggregate[baseline]["stimulus_decoding"]["generated"][target]["balanced_accuracy"]["mean"]
            for target in stim_targets
        ]
        err = [
            aggregate[baseline]["stimulus_decoding"]["generated"][target]["balanced_accuracy"]["sem"]
            for target in stim_targets
        ]
        axes[1].bar(
            xpos + (idx - (len(baselines) - 1) / 2) * width,
            vals,
            width,
            yerr=err,
            color=colors[baseline],
            capsize=3,
            label=labels[baseline],
        )
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(stim_labels)
    if len(stim_targets) == 1:
        axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Stimulus decode balanced accuracy")
    axes[1].set_title("Stimulus preservation")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=colors[baseline], edgecolor="none", label=labels[baseline])
        for baseline in baselines
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#222222", linestyle=":", linewidth=2.5, label="Source spikes"),
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.4, label="Chance"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fold_indices(args, cfg):
    if args.fold_indices is not None:
        return [int(idx) for idx in args.fold_indices]
    return list(range(int(getattr(cfg, "cv_folds", 1)) if args.cv else 1))


FIXED_ARGS = {
    "noreg_config": "configs/allen_vc/decode_stim_speed_orientation_noreg_cv10.yaml",
    "gaussian_glm_config": "configs/allen_vc/gaussian-glm-cv10.yaml",
    "cv": True,
    "cv_folds": 10,
    "seed": 42,
    "batch_size": 512,
    "loader_batch_size": 4096,
    "num_workers": 0,
    "speed_decoder": "mlp",
    "decoder_hidden_dim": 256,
    "decoder_num_layers": 3,
    "decoder_activation": "gelu",
    "decoder_dropout": 0.0,
    "decoder_lr": 1e-3,
    "decoder_weight_decay": 1e-4,
    "decoder_max_epochs": 300,
    "decoder_patience": 20,
    "allen_spike_smoothing": 2.0,
    "generated_decode_features": "poisson_smooth",
    "pivae_latent_dim": 4,
    "pivae_gen_nodes": 60,
    "pivae_flow_blocks": 2,
    "pivae_lr": 5e-4,
    "pivae_epochs": 600,
    "pivae_patience": 40,
    "pivae_batch_size": 256,
    "pivae_samples": 32,
    "neds_root": "NEDS",
    "if_cuda": True,
    # NEDS settings mirror NEDS/src/configs/multi_modal/{trainer_mm,mm_single_session}.yaml,
    # except epochs, learning rate and patience, which are the paper's Allen settings.
    # NEDS runs in encoding mode for the counterfactual baseline because it predicts spikes from
    # provided stimulus/behaviour sequences in this evaluation. Its max_F is set per fold to the
    # presentation length.
    "neds_epochs": 4000,
    "neds_sequence_batch_size": 16,
    "neds_predict_batch_size": 1024,
    "neds_lr": 5e-4,
    "neds_weight_decay": 0.01,
    "neds_hidden_size": 256,
    "neds_inter_size": 512,
    "neds_layers": 5,
    "neds_embedder_dropout": 0.2,
    "neds_transformer_dropout": 0.4,
    "neds_patience": 50,
    "neds_min_delta": 1e-4,
    "neds_verbose": False,
    "result_name": None,
    "figure_name": None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["neural_disco", "no_reg", "gaussian_glm", "pivae", "neds"],
        default=["pivae", "neds"],
    )
    parser.add_argument("--fold-indices", nargs="+", type=int, default=None)
    parser.add_argument("--force-baseline-retrain", action="store_true")
    args = parser.parse_args()
    for key, value in FIXED_ARGS.items():
        setattr(args, key, value)

    cfg = load_config(args.config)
    if args.cv_folds is not None:
        cfg.cv_folds = args.cv_folds
    cfg_behaviours = list(cfg.behaviours)
    if "speed" not in cfg_behaviours or "temporal_frequency" in cfg_behaviours:
        raise ValueError(
            "This Allen baseline script requires speed as a known confound and holds out "
            f"temporal_frequency. Got cfg.behaviours={cfg_behaviours}."
        )
    noreg_cfg = None
    if "no_reg" in args.baselines:
        noreg_cfg = load_config(args.noreg_config)
        noreg_behaviours = list(noreg_cfg.behaviours)
        if "speed" not in noreg_behaviours or "temporal_frequency" in noreg_behaviours:
            raise ValueError(
                "w/o reg. Allen speed-to-mean config requires speed and must hold out "
                f"temporal_frequency. Got cfg.behaviours={noreg_behaviours}."
            )
        if args.cv_folds is not None:
            noreg_cfg.cv_folds = args.cv_folds
    gaussian_glm_cfg = None
    if "gaussian_glm" in args.baselines:
        gaussian_glm_cfg = load_config(args.gaussian_glm_config)
        gaussian_behaviours = list(gaussian_glm_cfg.behaviours)
        if (
            "speed" not in gaussian_behaviours
            or "temporal_frequency" in gaussian_behaviours
        ):
            raise ValueError(
                "Gaussian GLM Allen speed-to-mean config requires speed and "
                "must hold out temporal_frequency. "
                f"Got cfg.behaviours={gaussian_behaviours}."
            )
        if args.cv_folds is not None:
            gaussian_glm_cfg.cv_folds = args.cv_folds

    log_dir = get_log_dir(cfg, args.config)

    fold_results = []
    for fold_idx in _fold_indices(args, cfg):
        print(f"=== Allen speed-to-mean baseline fold {fold_idx} ===", flush=True)
        datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx)
        train_dataset = datamodule.train_dataset if args.cv else datamodule.dataset("train")
        train_batch = collate_all_batches(_loader(train_dataset, args.loader_batch_size))
        test_batch = collate_all_batches(_loader(datamodule.dataset("test"), args.loader_batch_size))
        train_spikes = gaussian_filter1d(
            _flatten_spikes(train_batch["spikes"]),
            sigma=args.allen_spike_smoothing,
            axis=0,
        )
        test_spikes = gaussian_filter1d(
            _flatten_spikes(test_batch["spikes"]),
            sigma=args.allen_spike_smoothing,
            axis=0,
        )
        if "neural_disco" in args.baselines:
            fold_results.append(
                _run_neural_disco_fold(
                    cfg,
                    args.config,
                    "neural_disco",
                    args,
                    fold_idx,
                    train_batch,
                    test_batch,
                    train_spikes,
                    test_spikes,
                )
            )
        if "no_reg" in args.baselines:
            fold_results.append(
                _run_neural_disco_fold(
                    noreg_cfg,
                    args.noreg_config,
                    "no_reg",
                    args,
                    fold_idx,
                    train_batch,
                    test_batch,
                    train_spikes,
                    test_spikes,
                )
            )
        if "gaussian_glm" in args.baselines:
            gaussian_datamodule = load_datamodule_from_config(
                gaussian_glm_cfg,
                fold_idx=fold_idx,
            )
            gaussian_train_dataset = (
                gaussian_datamodule.train_dataset
                if args.cv
                else gaussian_datamodule.dataset("train")
            )
            gaussian_train_batch = collate_all_batches(
                _loader(gaussian_train_dataset, args.loader_batch_size)
            )
            gaussian_test_batch = collate_all_batches(
                _loader(
                    gaussian_datamodule.dataset("test"),
                    args.loader_batch_size,
                )
            )
            fold_results.append(
                _run_gaussian_glm_fold(
                    gaussian_glm_cfg,
                    args.gaussian_glm_config,
                    args,
                    fold_idx,
                    gaussian_train_batch,
                    gaussian_test_batch,
                    train_batch,
                    test_batch,
                    train_spikes,
                    test_spikes,
                )
            )
        if "pivae" in args.baselines:
            fold_results.append(
                _run_pivae_fold(cfg, args, fold_idx, log_dir, datamodule, train_batch, test_batch, train_spikes, test_spikes)
            )
        if "neds" in args.baselines:
            fold_results.append(
                _run_neds_fold(cfg, args, fold_idx, log_dir, datamodule, train_batch, test_batch, train_spikes, test_spikes)
            )

    aggregate = _aggregate(fold_results)
    result = {
        "config": args.config,
        "dataset": cfg.dataset,
        "cv": bool(args.cv),
        "cv_folds": int(getattr(cfg, "cv_folds", 1)) if args.cv else None,
        "fold_indices": _fold_indices(args, cfg),
        "speed_mean_source": "train",
        "generated_decode_features": args.generated_decode_features,
        "speed_decoder": args.speed_decoder,
        "baselines": args.baselines,
        "noreg_config": args.noreg_config if "no_reg" in args.baselines else None,
        "gaussian_glm_config": (
            args.gaussian_glm_config
            if "gaussian_glm" in args.baselines
            else None
        ),
        "fold_results": fold_results,
        "aggregate": aggregate,
    }
    fold_tag = "folds_" + "_".join(map(str, result["fold_indices"]))
    stem = f"allen_speed_mean_baselines_{fold_tag}"
    result_path = get_result_filepath(log_dir, args.result_name or f"{stem}_metrics.json")
    figure_path = get_visualization_filepath(log_dir, args.figure_name or f"{stem}.png")
    _plot_summary(aggregate, figure_path)
    result["figure_path"] = figure_path
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"result_path": result_path, "figure_path": figure_path, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
