import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, Subset

# MC Maze NEDS baseline only. Allen NEDS runs live in
# allen_speed_mean_baselines.py because Allen must be grouped into complete
# drifting-grating presentation sequences before NEDS sees the data.
current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.cfact.mc_maze_mlp_decode_xy import (  # noqa: E402
    _ci95,
    _normalize_between_controls,
    build_relaxed_pairs,
    build_trial_windows,
    decoder_predict,
    generated_activity_to_decoder_features,
    prepare_behaviour_for_model_input,
)
from scripts.cfact.mc_maze_pairing import PAIRING_STRATEGY  # noqa: E402
from scripts.cfact.pivae_behaviour_only_baseline import (  # noqa: E402
    _flatten_spikes,
    _loader,
    _mc_fit_decoders,
    _pooled_r2,
    bits_per_spike,
    set_seed,
)
from src.data.collate import collate_all_batches  # noqa: E402
from src.eval.mc_maze_windows import (  # noqa: E402
    window_condition_spike_target,
    window_condition_target,
)
from src.plotting import COUNTERFACTUAL_SUMMARY_COLORS  # noqa: E402
from src.utils import (  # noqa: E402
    get_log_dir,
    get_result_filepath,
    get_visualization_filepath,
    load_config,
    load_datamodule_from_config,
)


def _load_neds_modules(neds_root):
    neds_src = Path(neds_root).expanduser().resolve() / "src"
    if not neds_src.exists():
        raise FileNotFoundError(f"NEDS src directory not found: {neds_src}")
    neds_src_str = str(neds_src)
    if neds_src_str not in sys.path:
        sys.path.insert(0, neds_src_str)

    from multi_modal import encoder_embeddings as neds_encoder_module  # noqa: E402
    from multi_modal import mm as neds_mm_module  # noqa: E402
    from multi_modal.encoder_embeddings import EncoderEmbedding  # noqa: E402
    from multi_modal.mm import MultiModal  # noqa: E402
    from models import stitcher as neds_stitcher_module  # noqa: E402
    from utils.config_utils import update_config  # noqa: E402

    return neds_encoder_module, neds_mm_module, neds_stitcher_module, EncoderEmbedding, MultiModal, update_config


def _behaviour_specs(cfg):
    specs = []
    dims_cfg = getattr(cfg, "behaviour_dims", {})
    for behaviour in cfg.behaviours:
        dim = int(dims_cfg.get(behaviour, 1)) if hasattr(dims_cfg, "get") else 1
        for dim_idx in range(dim):
            if getattr(cfg, "dataset", None) == "allen_vc" and behaviour == "speed" and dim_idx == 0:
                # NEDS' IBL code maps wheel-speed to the dynamic "wheel" modality.
                modality = "wheel"
            else:
                modality = f"{behaviour}_{dim_idx}" if dim > 1 else behaviour
            specs.append((behaviour, dim_idx, modality))
    return specs


def _prepare_neds_config(args, update_config):
    config = update_config(str(Path(args.neds_root) / "src/configs/multi_modal/mm_single_session.yaml"))
    config["masker"]["force_active"] = True
    config["masker"]["zero_ratio"] = 1.0
    config["model"] = config
    config["model"]["encoder"]["embedder"]["n_modality"] = 1 + int(args.neds_num_modalities)
    config["model"]["encoder"]["embedder"]["max_F"] = int(args.neds_max_F)
    config["model"]["encoder"]["embedder"]["dropout"] = args.neds_embedder_dropout
    config["model"]["encoder"]["transformer"]["hidden_size"] = args.neds_hidden_size
    config["model"]["encoder"]["transformer"]["inter_size"] = args.neds_inter_size
    config["model"]["encoder"]["transformer"]["n_layers"] = args.neds_layers
    config["model"]["encoder"]["transformer"]["dropout"] = args.neds_transformer_dropout
    return config


def _build_neds_model(args, cfg, num_neurons, device):
    (
        neds_encoder_module,
        neds_mm_module,
        neds_stitcher_module,
        EncoderEmbedding,
        MultiModal,
        update_config,
    ) = _load_neds_modules(args.neds_root)
    specs = _behaviour_specs(cfg)
    args.neds_num_modalities = len(specs)
    if args.neds_max_F is None:
        args.neds_max_F = int(getattr(cfg, "input_window", 1))
    neds_cfg = _prepare_neds_config(args, update_config)
    eid = str(cfg.dataset)
    eid_list = {eid: int(num_neurons)}
    neds_encoder_module.INCLUDE_EIDS = [eid]
    behaviour_modalities = {spec[2] for spec in specs}
    neds_mm_module.DYNAMIC_VARS = sorted(set(neds_mm_module.DYNAMIC_VARS) | behaviour_modalities)
    neds_stitcher_module.DYNAMIC_VARS = sorted(set(neds_stitcher_module.DYNAMIC_VARS) | behaviour_modalities)
    for modality in behaviour_modalities:
        neds_stitcher_module.OUTPUT_DIM.setdefault(modality, 1)

    static_behaviour_encoder = bool(getattr(args, "neds_static_behaviour_encoder", False))
    encoder_embeddings = {}
    for _, _, modality in specs:
        use_static_encoder = (
            static_behaviour_encoder
            and modality not in {"wheel", "whisker", "choice", "block"}
        )
        if use_static_encoder:
            encoder_embeddings[modality] = EncoderEmbedding(
                n_channel=1,
                output_channel=1,
                config=neds_cfg.model.encoder,
                stitching=False,
                eid_list=eid_list,
                mod=modality,
                max_F=int(args.neds_max_F),
            )
        else:
            encoder_embeddings[modality] = EncoderEmbedding(
                hidden_size=args.neds_hidden_size,
                n_channel=args.neds_hidden_size,
                output_channel=args.neds_hidden_size,
                config=neds_cfg.model.encoder,
                stitching=True,
                eid_list=eid_list,
                mod=modality,
                max_F=int(args.neds_max_F),
            )

    model = MultiModal(
        encoder_embeddings,
        ["spike"] + [spec[2] for spec in specs],
        [spec[2] for spec in specs],
        "encoding",
        neds_cfg.model,
        eid_list=eid_list,
    )
    if not static_behaviour_encoder:
        for _, _, modality in specs:
            model.mod_type[modality] = "dynamic"
            model.num_class.setdefault(modality, None)
    if isinstance(getattr(model, "mod_stitcher_proj_dict", None), dict):
        # NEDS constructs this in a plain dict in encoding mode. Registering it
        # preserves the author architecture but makes PyTorch train/checkpoint
        # the spike readout instead of leaving it as an unsaved random module.
        model.mod_stitcher_proj_dict = torch.nn.ModuleDict(model.mod_stitcher_proj_dict)
    return model.to(device), specs, eid


def _fit_neds_input_normalizer(dataset, behaviours, batch_size):
    batch = collate_all_batches(_loader(dataset, batch_size))
    normalizer = {}
    for behaviour in behaviours:
        value = batch[behaviour].detach().cpu().numpy()
        if value.ndim == 3:
            value = value.reshape(-1, value.shape[-1])
            keep_shape = (1, 1, value.shape[-1])
        else:
            value = value.reshape(len(batch[behaviour]), -1)
            keep_shape = (1, value.shape[-1])
        min_val = np.nanmin(value, axis=0, keepdims=True).astype(np.float32).reshape(keep_shape)
        max_val = np.nanmax(value, axis=0, keepdims=True).astype(np.float32).reshape(keep_shape)
        scale = np.maximum(max_val - min_val, 1e-6).astype(np.float32)
        normalizer[behaviour] = {"min": min_val, "scale": scale}
    return normalizer


def _normalize_behaviour_value(value, behaviour, normalizer):
    if normalizer is None or behaviour not in normalizer:
        return value
    is_tensor = torch.is_tensor(value)
    value_np = value.detach().cpu().numpy() if is_tensor else np.asarray(value)
    original_shape = value_np.shape
    norm = normalizer[behaviour]
    normalized = ((value_np.astype(np.float32) - norm["min"]) / norm["scale"]).reshape(original_shape).astype(np.float32)
    if is_tensor:
        return torch.from_numpy(normalized).to(value.device)
    return normalized


def _normalize_batch_for_neds(batch, specs, normalizer):
    if normalizer is None:
        return batch
    normalized = dict(batch)
    for behaviour, _, _ in specs:
        normalized[behaviour] = _normalize_behaviour_value(batch[behaviour], behaviour, normalizer)
    return normalized


def _normalize_behaviour_arrays_for_neds(behaviour_arrays, normalizer):
    if normalizer is None:
        return behaviour_arrays
    return {
        behaviour: _normalize_behaviour_value(value, behaviour, normalizer)
        for behaviour, value in behaviour_arrays.items()
    }


def _serializable_normalizer(normalizer):
    if normalizer is None:
        return None
    return {
        behaviour: {
            "min": values["min"].reshape(-1).astype(float).tolist(),
            "scale": values["scale"].reshape(-1).astype(float).tolist(),
        }
        for behaviour, values in normalizer.items()
    }


def _resolve_device(args):
    if args.if_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("NEDS requested CUDA, but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
        print(f"Using NEDS device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
        return device

    device = torch.device("cpu")
    print("Using NEDS device: cpu", flush=True)
    return device


def _spikes_to_neds_targets(spikes):
    spikes = torch.as_tensor(spikes).float()
    if spikes.dim() == 2:
        return spikes.unsqueeze(1)
    if spikes.dim() == 3 and spikes.shape[-1] != 1 and spikes.shape[1] != 1:
        return spikes.contiguous()
    if spikes.dim() == 3 and spikes.shape[-1] == 1:
        return spikes.transpose(1, 2).contiguous()
    if spikes.dim() == 3 and spikes.shape[1] == 1:
        return spikes.contiguous()
    raise ValueError(f"Unsupported spike tensor shape for NEDS: {tuple(spikes.shape)}")


def _behaviour_to_neds_input(value, dim_idx):
    value = torch.as_tensor(value).float()
    if value.dim() == 1:
        value = value[:, None]
    if value.dim() == 2:
        return value[:, dim_idx : dim_idx + 1].unsqueeze(1).contiguous()
    if value.dim() == 3:
        return value[:, :, dim_idx : dim_idx + 1].contiguous()
    raise ValueError(f"Unsupported behaviour tensor shape for NEDS: {tuple(value.shape)}")


def _neds_mod_dict_from_batch(
    batch,
    specs,
    device,
    eid,
    num_neurons=None,
    spike_targets=None,
    normalizer=None,
):
    batch = _normalize_batch_for_neds(batch, specs, normalizer)
    if spike_targets is None:
        spike_targets = _spikes_to_neds_targets(batch["spikes"])
    else:
        spike_targets = _spikes_to_neds_targets(spike_targets)
    spike_targets = spike_targets.to(device)
    batch_size = spike_targets.shape[0]
    time_steps = spike_targets.shape[1]
    num_neurons = int(num_neurons or spike_targets.shape[-1])

    time_mask = torch.ones((batch_size, time_steps), dtype=torch.int64, device=device)
    timestamps = torch.arange(time_steps, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
    spike_mask = torch.ones_like(spike_targets, dtype=torch.int64, device=device)
    zero_mask = torch.zeros_like(spike_targets, dtype=torch.int64, device=device)

    mod_dict = {
        "spike": {
            "inputs_modality": torch.tensor(0, device=device),
            "targets_modality": torch.tensor(0, device=device),
            "inputs_attn_mask": time_mask,
            "inputs_timestamp": timestamps,
            "targets_timestamp": timestamps,
            "eid": [eid] * batch_size,
            "num_neuron": num_neurons,
            "training_mode": "encoding",
            "inputs": spike_targets,
            "targets": spike_targets,
            "eval_mask": spike_mask,
        }
    }

    for mod_idx, (behaviour, dim_idx, modality) in enumerate(specs, start=1):
        x = _behaviour_to_neds_input(batch[behaviour], dim_idx).to(device)
        mod_dict[modality] = {
            "inputs_modality": torch.tensor(mod_idx, device=device),
            "targets_modality": torch.tensor(mod_idx, device=device),
            "inputs_attn_mask": time_mask,
            "inputs_timestamp": timestamps,
            "targets_timestamp": timestamps,
            "eid": [eid] * batch_size,
            "num_neuron": num_neurons,
            "training_mode": "encoding",
            "inputs": x,
            "targets": x,
            "eval_mask": zero_mask[:, :, :1],
        }
    return mod_dict


def _neds_mod_dict_from_behaviour_arrays(behaviour_arrays, specs, device, eid, num_neurons, normalizer=None):
    behaviour_arrays = _normalize_behaviour_arrays_for_neds(behaviour_arrays, normalizer)
    first = next(iter(behaviour_arrays.values()))
    first = np.asarray(first)
    batch_size = int(first.shape[0])
    time_steps = int(first.shape[1]) if first.ndim == 3 else 1
    zero_spikes = torch.zeros((batch_size, time_steps, int(num_neurons)), dtype=torch.float32)
    batch = {name: torch.as_tensor(value).float() for name, value in behaviour_arrays.items()}
    batch["spikes"] = zero_spikes
    # This is the same multimodal encoding convention used by NEDS eval:
    # spike tokens are fully masked and behaviour tokens remain visible.
    return _neds_mod_dict_from_batch(
        batch,
        specs,
        device,
        eid,
        num_neurons=num_neurons,
        normalizer=None,
    )


def _predict_neds_from_behaviour(model, behaviour_arrays, specs, device, eid, num_neurons, batch_size, normalizer=None):
    model.eval()
    n = int(np.asarray(next(iter(behaviour_arrays.values()))).shape[0])
    pred_batches = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            chunk = {name: np.asarray(value)[start:stop] for name, value in behaviour_arrays.items()}
            outputs = model(_neds_mod_dict_from_behaviour_arrays(chunk, specs, device, eid, num_neurons, normalizer))
            pred_batches.append(torch.exp(outputs.mod_preds["spike"]).detach().cpu())
    return torch.cat(pred_batches, dim=0).squeeze(1).numpy()


def _predict_neds_dataset(model, dataloader, specs, device, eid, normalizer=None):
    model.eval()
    pred_batches = []
    target_batches = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(_neds_mod_dict_from_batch(batch, specs, device, eid, normalizer=normalizer))
            pred_batches.append(torch.exp(outputs.mod_preds["spike"]).detach().cpu())
            target_batches.append(outputs.mod_targets["spike"].detach().cpu())
    preds = torch.cat(pred_batches, dim=0).squeeze(1).numpy()
    targets = torch.cat(target_batches, dim=0).squeeze(1).numpy()
    return preds, targets


def _train_neds(model, train_loader, val_loader, specs, device, eid, args, log_dir, fold_idx, normalizer=None):
    norm_tag = "_norm"
    seq_len = int(args.neds_max_F or 1)
    seq_tag = f"_seq{seq_len}" if seq_len > 1 else ""
    arch_tag = "_nedsdefault"
    readout_tag = "_registeredout"
    mode_tag = "_encoding_encoding"
    checkpoint_name = (
        f"neds{mode_tag}{norm_tag}{seq_tag}{arch_tag}{readout_tag}_fold_{fold_idx}.pt"
        if args.cv
        else f"neds{mode_tag}{norm_tag}{seq_tag}{arch_tag}{readout_tag}.pt"
    )
    checkpoint_path = Path(get_result_filepath(f"{log_dir}/fold_{fold_idx}" if args.cv else log_dir, checkpoint_name))
    if checkpoint_path.exists() and not args.force_baseline_retrain:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        return {
            "checkpoint_path": str(checkpoint_path),
            "best_epoch": int(state.get("epoch", -1)),
            "best_val_loss": float(state.get("val_loss", float("nan"))),
            "early_stop_patience": int(state.get("early_stop_patience", args.neds_patience)),
            "early_stop_min_delta": float(state.get("early_stop_min_delta", args.neds_min_delta)),
            "history": state.get("history", []),
            "loaded_from_cache": True,
        }

    optimizer = AdamW(model.parameters(), lr=args.neds_lr, weight_decay=args.neds_weight_decay, eps=1e-8)
    total_steps = max(1, args.neds_epochs * len(train_loader))
    scheduler = OneCycleLR(
        optimizer=optimizer,
        total_steps=total_steps,
        max_lr=args.neds_lr,
        pct_start=0.15,
        div_factor=10,
        anneal_strategy="cos",
    )
    best_val_loss = math.inf
    best_epoch = -1
    best_state = None
    history = []
    patience_ctr = 0

    for epoch in range(args.neds_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                _neds_mod_dict_from_batch(
                    batch,
                    specs,
                    device,
                    eid,
                    normalizer=normalizer,
                )
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    _neds_mod_dict_from_batch(
                        batch,
                        specs,
                        device,
                        eid,
                        normalizer=normalizer,
                    )
                )
                val_losses.append(float(outputs.loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": int(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if args.neds_verbose:
            print(f"NEDS fold {fold_idx} epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        improved = val_loss < (best_val_loss - args.neds_min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "epoch": int(best_epoch),
                    "val_loss": float(best_val_loss),
                    "early_stop_patience": int(args.neds_patience),
                    "early_stop_min_delta": float(args.neds_min_delta),
                    "history": history,
                },
                checkpoint_path,
            )
            patience_ctr = 0
        elif args.neds_patience > 0:
            patience_ctr += 1
            if patience_ctr >= args.neds_patience:
                break

    if best_state is None:
        raise RuntimeError("NEDS training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "early_stop_patience": int(args.neds_patience),
        "early_stop_min_delta": float(args.neds_min_delta),
        "history": history,
        "loaded_from_cache": False,
    }


def _bootstrap_neds_pair_metrics(pair_arrays, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    metrics = {
        "source_to_target_r2": _pooled_r2(pair_arrays, "target_true", "source_decoded"),
        "target_to_target_r2": _pooled_r2(pair_arrays, "target_true", "target_decoded"),
        "neds_to_source_r2": _pooled_r2(pair_arrays, "source_true", "neds_decoded"),
        "neds_to_target_r2": _pooled_r2(pair_arrays, "target_true", "neds_decoded"),
    }
    metrics["neds_gain_to_target"] = metrics["neds_to_target_r2"] - metrics["source_to_target_r2"]
    metrics["neds_gap_closed"] = _normalize_between_controls(
        metrics["neds_to_target_r2"],
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
        neds_to_source = _pooled_r2(rows, "source_true", "neds_decoded")
        neds_to_target = _pooled_r2(rows, "target_true", "neds_decoded")
        boot["source_to_target_r2"].append(source_to_target)
        boot["target_to_target_r2"].append(target_to_target)
        boot["neds_to_source_r2"].append(neds_to_source)
        boot["neds_to_target_r2"].append(neds_to_target)
        boot["neds_gain_to_target"].append(neds_to_target - source_to_target)
        boot["neds_gap_closed"].append(
            _normalize_between_controls(neds_to_target, source_to_target, target_to_target)
        )
    for key, vals in boot.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        metrics[f"{key}_ci95"] = _ci95(vals) if len(vals) else float("nan")
    return metrics


class MCMazeReachSequenceDataset(Dataset):
    """Full-reach sequence view over an NLB fold subset."""

    def __init__(self, dataset, cfg):
        if int(cfg.output_window) != 1:
            raise ValueError("MC Maze NEDS sequence baseline expects output_window=1.")

        base_dataset, sample_indices = self._unwrap_subset(dataset)
        if not hasattr(base_dataset, "data_loader"):
            raise ValueError("MC Maze NEDS sequence wrapper requires an NLBDataset-like base dataset.")

        self.base_dataset = base_dataset
        self.behaviours = list(cfg.behaviours)
        self.input_window = int(cfg.input_window)
        self.output_window = int(cfg.output_window)
        self.spike_target_mode = cfg.spike_target_mode if "spike_target_mode" in cfg else None
        self.behaviour_target_mode = (
            cfg.behaviour_target_mode
            if "behaviour_target_mode" in cfg
            else (self.spike_target_mode or "future")
        )
        self.raw_spikes = base_dataset.data_loader.load_spikes().astype(np.float32)
        self.raw_behaviours = {
            behaviour: base_dataset.data_loader.load_behaviour(behaviour).astype(np.float32)
            for behaviour in self.behaviours
        }

        n_trials = int(self.raw_spikes.shape[0])
        if len(base_dataset) % n_trials != 0:
            raise ValueError(
                f"Cannot map {len(base_dataset)} flattened NLB samples to {n_trials} full trials."
            )
        self.seq_len = len(base_dataset) // n_trials

        grouped_bins = {}
        for sample_idx in np.asarray(sample_indices, dtype=np.int64):
            trial_idx = int(sample_idx // self.seq_len)
            bin_idx = int(sample_idx % self.seq_len)
            grouped_bins.setdefault(trial_idx, []).append(bin_idx)

        expected_bins = list(range(self.seq_len))
        self.trial_indices = []
        for trial_idx in sorted(grouped_bins):
            bins = sorted(grouped_bins[trial_idx])
            if bins != expected_bins:
                raise ValueError(
                    "MC Maze NEDS sequence baseline received a partial trial. "
                    f"trial={trial_idx}, expected_bins={expected_bins}, observed_bins={bins}"
                )
            self.trial_indices.append(trial_idx)

        if not self.trial_indices:
            raise ValueError("MC Maze NEDS sequence dataset has no full trials.")

    @staticmethod
    def _unwrap_subset(dataset):
        indices = None
        base = dataset
        while isinstance(base, Subset):
            subset_indices = np.asarray(base.indices, dtype=np.int64)
            indices = subset_indices if indices is None else subset_indices[indices]
            base = base.dataset
        if indices is None:
            indices = np.arange(len(base), dtype=np.int64)
        return base, indices

    def __len__(self):
        return len(self.trial_indices)

    def __getitem__(self, idx):
        trial_idx = int(self.trial_indices[idx])
        spikes = window_condition_spike_target(
            self.raw_spikes[trial_idx : trial_idx + 1],
            input_window=self.input_window,
            output_window=self.output_window,
            delay_factor=0,
            target_mode=(self.spike_target_mode or "future"),
        )
        if spikes.ndim != 3 or spikes.shape[0] != self.seq_len:
            raise ValueError(f"Unexpected MC Maze spike sequence shape: {spikes.shape}")
        if spikes.shape[-1] != 1:
            raise ValueError(f"MC Maze NEDS sequence baseline expects one-bin spike targets, got {spikes.shape}")
        item = {
            "spikes": torch.from_numpy(spikes[:, :, 0].astype(np.float32)),
            "trial_idx": torch.tensor(trial_idx, dtype=torch.long),
        }
        for behaviour in self.behaviours:
            values = window_condition_target(
                self.raw_behaviours[behaviour][trial_idx : trial_idx + 1],
                input_window=self.input_window,
                output_window=self.output_window,
                delay_factor=0,
                target_mode=self.behaviour_target_mode,
            )
            values = prepare_behaviour_for_model_input(self.base_dataset, behaviour, values)
            if values.shape[0] != self.seq_len:
                raise ValueError(
                    f"Unexpected MC Maze behaviour sequence shape for {behaviour}: {values.shape}"
                )
            item[behaviour] = torch.from_numpy(values.astype(np.float32))
        return item


def _make_train_val_loaders(cfg, args, datamodule, fold_idx):
    fold_train_dataset = datamodule.train_dataset if args.cv else datamodule.dataset("train")
    fold_val_dataset = datamodule.val_dataset if args.cv else datamodule.train_dataset
    if cfg.dataset == "mc_maze":
        fold_train_dataset = MCMazeReachSequenceDataset(fold_train_dataset, cfg)
        fold_val_dataset = MCMazeReachSequenceDataset(fold_val_dataset, cfg)
        if args.neds_max_F is None:
            args.neds_max_F = fold_train_dataset.seq_len
        elif int(args.neds_max_F) != fold_train_dataset.seq_len:
            raise ValueError(
                f"NEDS max_F={args.neds_max_F} does not match MC Maze reach length "
                f"{fold_train_dataset.seq_len}."
            )
    train_loader = torch.utils.data.DataLoader(
        fold_train_dataset,
        batch_size=args.neds_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        fold_val_dataset,
        batch_size=args.neds_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def _mc_fold(cfg, args, fold_idx, log_dir, decoders=None):
    set_seed(args.seed + fold_idx)
    datamodule = load_datamodule_from_config(cfg, fold_idx=fold_idx)
    eval_dataset = datamodule.dataset("test")
    num_neurons = int(eval_dataset[0]["spikes"].shape[0])
    device = _resolve_device(args)
    train_loader, val_loader = _make_train_val_loaders(cfg, args, datamodule, fold_idx)
    model, specs, eid = _build_neds_model(args, cfg, num_neurons, device)
    normalizer = _fit_neds_input_normalizer(train_loader.dataset, cfg.behaviours, args.loader_batch_size)
    training = _train_neds(model, train_loader, val_loader, specs, device, eid, args, log_dir, fold_idx, normalizer)

    if decoders is None:
        decoders = _mc_fit_decoders(cfg, args, datamodule)

    trial_info = eval_dataset.data_loader.load_trial_info().reset_index(drop=True)
    pairs = build_relaxed_pairs(trial_info)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("No MC Maze relaxed pairs available for NEDS evaluation.")

    pair_records = []
    target_behaviour_batches = {behaviour: [] for behaviour in cfg.behaviours}
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
        for behaviour in cfg.behaviours:
            target_behaviour_batches[behaviour].append(
                prepare_behaviour_for_model_input(eval_dataset, behaviour, target_behaviour[behaviour])[None, ...]
            )
        pair_records.append(
            {
                "source_inputs": source_inputs,
                "target_inputs": target_inputs,
                "source_behaviour": source_behaviour,
                "target_behaviour": target_behaviour,
            }
        )

    target_behaviour_cat = {
        behaviour: np.concatenate(values, axis=0).astype(np.float32)
        for behaviour, values in target_behaviour_batches.items()
    }
    neds_rates_cat = _predict_neds_from_behaviour(
        model,
        target_behaviour_cat,
        specs,
        device,
        eid,
        num_neurons,
        args.neds_predict_batch_size,
        normalizer,
    )
    if neds_rates_cat.ndim != 3:
        raise ValueError(f"Expected sequence NEDS rates with shape [pairs, time, neurons], got {neds_rates_cat.shape}")

    per_pair_arrays = {behaviour: [] for behaviour in cfg.behaviours}
    target_spikes_all = []
    neds_rates_all = []
    for idx, record in enumerate(pair_records):
        neds_rates = neds_rates_cat[idx]
        neds_decode_features = generated_activity_to_decoder_features(
            neds_rates,
            mode=args.mc_generated_decode_features,
            seed=args.seed + fold_idx * 1000003 + idx,
        )
        source_spikes = _flatten_spikes(record["source_inputs"]["spikes"])
        target_spikes = _flatten_spikes(record["target_inputs"]["spikes"])
        target_spikes_all.append(target_spikes)
        neds_rates_all.append(neds_rates)
        for behaviour in cfg.behaviours:
            decoder = decoders[behaviour]
            per_pair_arrays[behaviour].append(
                {
                    "source_true": record["source_behaviour"][behaviour],
                    "target_true": record["target_behaviour"][behaviour],
                    "source_decoded": decoder_predict(decoder, source_spikes),
                    "target_decoded": decoder_predict(decoder, target_spikes),
                    "neds_decoded": decoder_predict(decoder, neds_decode_features),
                }
            )

    summary = {
        behaviour: _bootstrap_neds_pair_metrics(
            per_pair_arrays[behaviour],
            n_bootstrap=args.summary_bootstrap_samples,
            seed=args.seed + fold_idx,
        )
        for behaviour in cfg.behaviours
    }
    summary["spikes"] = {
        "neds_bps_to_target_spikes": bits_per_spike(
            np.concatenate(neds_rates_all, axis=0),
            np.concatenate(target_spikes_all, axis=0),
        )
    }
    return {
        "fold_idx": fold_idx,
        "baseline_checkpoint": training["checkpoint_path"],
        "training": training,
        "input_normalizer": _serializable_normalizer(normalizer),
        "num_pairs": len(pairs),
        "pairing_strategy": PAIRING_STRATEGY,
        "summary": summary,
    }


def _aggregate_cv(fold_results):
    fold_summaries = [result["summary"] for result in fold_results]
    aggregate = {}
    for group in fold_summaries[0].keys():
        aggregate[group] = {}
        for metric in fold_summaries[0][group].keys():
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
    if cfg.dataset == "allen_vc":
        behaviours = [b for b in metrics if b != "spikes"]
        labels = [b.replace("_", " ").title() for b in behaviours]
        values = [metrics[b]["neds_to_counterfactual_target_r2"] for b in behaviours]
        yerr = None
        if cv and isinstance(values[0], dict):
            yerr = [metrics[b]["neds_to_counterfactual_target_r2"]["sem"] for b in behaviours]
            values = [metrics[b]["neds_to_counterfactual_target_r2"]["mean"] for b in behaviours]
        ylabel = r"Decode $R^2$ to prescribed counterfactual"
    else:
        behaviours = list(cfg.behaviours)
        labels = [b.replace("_", " ").title() for b in behaviours]
        values = [metrics[b]["neds_to_target_r2"] for b in behaviours]
        yerr = [metrics[b].get("neds_to_target_r2_ci95", 0.0) for b in behaviours]
        if cv and isinstance(values[0], dict):
            yerr = [metrics[b]["neds_to_target_r2"]["sem"] for b in behaviours]
            values = [metrics[b]["neds_to_target_r2"]["mean"] for b in behaviours]
        ylabel = r"Decode $R^2$ to target"

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=yerr, color=COUNTERFACTUAL_SUMMARY_COLORS["neds_baseline"], capsize=3)
    ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title("NEDS")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


FIXED_ARGS = {
    "cv": True,
    "cv_folds": 10,
    "max_cv_folds": None,
    "max_pairs": None,
    "batch_size": 512,
    "loader_batch_size": 4096,
    "num_workers": 0,
    "seed": 42,
    "if_cuda": True,
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
    "neds_root": "NEDS",
    # NEDS settings mirror NEDS/src/configs/multi_modal/{trainer_mm,mm_single_session}.yaml.
    # NEDS runs in encoding mode for the counterfactual baseline because it predicts spikes from
    # provided behaviour sequences in this evaluation.
    "neds_max_F": None,
    "neds_epochs": 2000,
    "neds_batch_size": 16,
    "neds_predict_batch_size": 1024,
    "neds_lr": 2e-4,
    "neds_weight_decay": 0.01,
    "neds_hidden_size": 256,
    "neds_inter_size": 512,
    "neds_layers": 5,
    "neds_embedder_dropout": 0.2,
    "neds_transformer_dropout": 0.4,
    "neds_patience": 0,
    "neds_min_delta": 1e-4,
    "neds_verbose": False,
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
    if cfg.dataset == "allen_vc":
        raise ValueError(
            "Run Allen NEDS through "
            "scripts/cfact/allen_speed_mean_baselines.py, which groups bins into "
            "complete drifting-grating presentation sequences. "
            "Do not use neds_behaviour_only_baseline.py for Allen."
        )
    if args.decoder == "auto":
        args.decoder = "cebra"

    log_dir = get_log_dir(cfg, args.config)
    num_folds = int(getattr(cfg, "cv_folds", 5)) if args.cv else 1
    fold_indices = list(range(num_folds))
    if args.max_cv_folds is not None:
        if args.max_cv_folds <= 0:
            raise ValueError("max_cv_folds must be positive.")
        fold_indices = fold_indices[: args.max_cv_folds]
    fold_results = []
    mc_decoders = None
    if args.cv and cfg.dataset == "mc_maze":
        decoder_datamodule = load_datamodule_from_config(cfg, fold_idx=0)
        mc_decoders = _mc_fit_decoders(cfg, args, decoder_datamodule)

    for fold_pos, fold_idx in enumerate(fold_indices):
        print(f"Evaluating NEDS fold {fold_pos + 1}/{len(fold_indices)} (fold_idx={fold_idx})", flush=True)
        if cfg.dataset == "mc_maze":
            fold_results.append(_mc_fold(cfg, args, fold_idx, log_dir, decoders=mc_decoders))
        else:
            raise ValueError(f"Unsupported dataset for NEDS: {cfg.dataset}")

    aggregate = _aggregate_cv(fold_results) if args.cv else fold_results[0]["summary"]
    stem = "neds_baseline_cv" if args.cv else "neds_baseline"
    result_path = get_result_filepath(log_dir, args.result_name or f"{stem}_metrics.json")
    figure_path = get_visualization_filepath(log_dir, args.figure_name or f"{stem}.png")
    result = {
        "config": args.config,
        "dataset": cfg.dataset,
        "cv": bool(args.cv),
        "cv_folds": len(fold_indices) if args.cv else None,
        "cv_folds_requested": num_folds if args.cv else None,
        "decoder": args.decoder,
        "mc_generated_decode_features": args.mc_generated_decode_features if cfg.dataset == "mc_maze" else None,
        "mc_pairing_strategy": PAIRING_STRATEGY if cfg.dataset == "mc_maze" else None,
        "baseline_type": "neds_encoding",
        "neds_root": str(Path(args.neds_root).expanduser().resolve()),
        "device": "cuda" if args.if_cuda else "cpu",
        "neds_hparams": {
            "epochs": args.neds_epochs,
            "model_mode": "encoding",
            "mixed_training": False,
            "mask_ratio": 0.3,
            "max_F": args.neds_max_F,
            "batch_size": args.neds_batch_size,
            "lr": args.neds_lr,
            "weight_decay": args.neds_weight_decay,
            "hidden_size": args.neds_hidden_size,
            "inter_size": args.neds_inter_size,
            "n_layers": args.neds_layers,
            "embedder_dropout": args.neds_embedder_dropout,
            "transformer_dropout": args.neds_transformer_dropout,
            "normalize_inputs": True,
            "patience": args.neds_patience,
            "min_delta": args.neds_min_delta,
            "seed": args.seed,
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
