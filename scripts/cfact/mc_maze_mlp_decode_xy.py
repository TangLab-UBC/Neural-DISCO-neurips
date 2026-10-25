import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

current_dir = os.getcwd()
sys.path.append(current_dir)

from src.eval.torch_decoder import TorchMLPDecoder
from src.eval.mc_maze_windows import (
    window_condition_array,
    window_condition_context_array,
    window_condition_spike_target,
    window_condition_target,
)
from src.data.collate import collate_all_batches
from src.modules.neural_disco_module import NeuralDISCOModule
from scripts.cfact.mc_maze_pairing import PAIRING_STRATEGY, build_relaxed_pairs


def _prepare_decoder_features(spikes):
    spikes = np.asarray(spikes, dtype=np.float32)
    if spikes.ndim <= 2:
        return spikes
    return spikes.reshape(spikes.shape[0], -1)


class CEBRAEmbeddingDecoder:
    def __init__(
        self,
        model_architecture="offset10-model",
        batch_size=1000,
        learning_rate=1e-4,
        temperature=1.0,
        output_dimension=8,
        max_iterations=15000,
        distance="cosine",
        conditional="time_delta",
        time_offsets=10,
        knn_neighbors=36,
        knn_metric="cosine",
        device="cuda_if_available",
        verbose=False,
        seed=0,
    ):
        self.model_architecture = model_architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.output_dimension = output_dimension
        self.max_iterations = max_iterations
        self.distance = distance
        self.conditional = conditional
        self.time_offsets = time_offsets
        self.knn_neighbors = knn_neighbors
        self.knn_metric = knn_metric
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def fit(self, X, y):
        import cebra

        X = _prepare_decoder_features(X)
        y = np.asarray(y, dtype=np.float32)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.model = cebra.CEBRA(
            model_architecture=self.model_architecture,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            output_dimension=self.output_dimension,
            max_iterations=self.max_iterations,
            distance=self.distance,
            conditional=self.conditional,
            device=self.device,
            verbose=self.verbose,
            time_offsets=self.time_offsets,
        )
        self.model.fit(X, y)
        embedding = self.model.transform(X)
        self.decoder = cebra.KNNDecoder(
            n_neighbors=self.knn_neighbors,
            metric=self.knn_metric,
        )
        self.decoder.fit(embedding, y)
        return self

    def predict(self, X):
        X = _prepare_decoder_features(X)
        embedding = self.model.transform(X)
        return np.asarray(self.decoder.predict(embedding), dtype=np.float32)


def fit_decoder(train_spikes, train_behaviour, decoder_type="cebra", hidden_dim=64, decoder_kwargs=None):
    decoder_kwargs = decoder_kwargs or {}
    train_spikes = _prepare_decoder_features(train_spikes)
    if decoder_type == "ridge":
        model = RidgeCV(alphas=np.logspace(-4, 4, 9))
        model.fit(train_spikes, train_behaviour)
        return model
    if decoder_type == "cebra":
        model = CEBRAEmbeddingDecoder(
            model_architecture=decoder_kwargs.get("cebra_model_architecture", "offset10-model"),
            batch_size=decoder_kwargs.get("cebra_batch_size", 1000),
            learning_rate=decoder_kwargs.get("cebra_learning_rate", 1e-4),
            temperature=decoder_kwargs.get("cebra_temperature", 1.0),
            output_dimension=decoder_kwargs.get("cebra_output_dimension", 8),
            max_iterations=decoder_kwargs.get("cebra_max_iterations", 15000),
            distance=decoder_kwargs.get("cebra_distance", "cosine"),
            conditional=decoder_kwargs.get("cebra_conditional", "time_delta"),
            time_offsets=decoder_kwargs.get("cebra_time_offsets", 10),
            knn_neighbors=decoder_kwargs.get("cebra_knn_neighbors", 36),
            knn_metric=decoder_kwargs.get("cebra_knn_metric", "cosine"),
            device=decoder_kwargs.get("cebra_device", "cuda_if_available"),
            verbose=decoder_kwargs.get("cebra_verbose", False),
            seed=decoder_kwargs.get("seed", 0),
        )
        model.fit(train_spikes, train_behaviour)
        return model
    if decoder_type != "mlp":
        raise ValueError(f"Unsupported decoder_type: {decoder_type}")
    model = TorchMLPDecoder(
        hidden_dim=hidden_dim,
        lr=decoder_kwargs.get("lr", 1e-3),
        weight_decay=decoder_kwargs.get("weight_decay", 1e-4),
        max_epochs=decoder_kwargs.get("max_epochs", 200),
        patience=decoder_kwargs.get("patience", 10),
        num_layers=decoder_kwargs.get("num_layers", 1),
        activation=decoder_kwargs.get("activation", "relu"),
        dropout=decoder_kwargs.get("dropout", 0.0),
        seed=decoder_kwargs.get("seed", 0),
    )
    model.fit(train_spikes, train_behaviour)
    return model


def decoder_predict(decoder, spikes):
    return decoder.predict(_prepare_decoder_features(spikes))


def generated_activity_to_decoder_features(activity, mode="rates", seed=0):
    activity = np.asarray(activity, dtype=np.float32)
    if mode == "rates":
        return activity
    if mode == "poisson":
        rng = np.random.default_rng(seed)
        return rng.poisson(np.clip(activity, 0.0, None)).astype(np.float32)
    raise ValueError(f"Unsupported generated decode feature mode: {mode}")


def batched_decoder_predict(decoder, spike_batches):
    if not spike_batches:
        return []
    lengths = [int(batch.shape[0]) for batch in spike_batches]
    cat = np.concatenate(spike_batches, axis=0)
    pred = decoder_predict(decoder, cat)
    offsets = np.cumsum([0] + lengths)
    return [pred[offsets[idx] : offsets[idx + 1]] for idx in range(len(lengths))]


def evaluate_reconstruction(module, val_dataset, device):
    loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    batch = collate_all_batches(loader)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        pred = module.model(batch)["reconstruction"].squeeze(-1)
    true = batch["spikes"].squeeze(-1)
    return {
        "val_bps": float(NeuralDISCOModule.bits_per_spike(pred, true).detach().cpu()),
        "val_r2": float(r2_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy())),
        "pred_mean": float(pred.mean().detach().cpu()),
        "true_mean": float(true.mean().detach().cpu()),
    }


def build_trial_windows(
    dataset,
    trial_idx,
    behaviours,
    input_window,
    output_window,
    unknown_input_window=None,
    spike_target_mode=None,
    behaviour_target_mode=None,
):
    spikes = dataset.data_loader.load_spikes()[trial_idx : trial_idx + 1]
    target_mode = spike_target_mode or "future"
    if spike_target_mode is None:
        spike_inputs = {
            "spikes": window_condition_array(
                spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=0,
                target_mode=target_mode,
            )
        }
    else:
        spike_inputs = {
            "spikes": window_condition_spike_target(
                spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=0,
                target_mode=target_mode,
            ),
            "spikes_input": window_condition_array(
                spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=0,
                target_mode=target_mode,
            ),
        }
    if unknown_input_window is not None and unknown_input_window > input_window:
        spike_inputs["spikes_context"] = window_condition_context_array(
            spikes,
            input_window=input_window,
            output_window=output_window,
            delay_factor=0,
            context_window=unknown_input_window,
            target_mode=target_mode,
        )
    behaviour_targets = {}
    behaviour_mode = behaviour_target_mode or spike_target_mode or "future"
    for behaviour in behaviours:
        values = dataset.data_loader.load_behaviour(behaviour)[trial_idx : trial_idx + 1]
        behaviour_targets[behaviour] = window_condition_target(
            values,
            input_window=input_window,
            output_window=output_window,
            delay_factor=0,
            target_mode=behaviour_mode,
        )
    return spike_inputs, behaviour_targets


def prepare_behaviour_for_model_input(dataset, behaviour, windows):
    mode = getattr(dataset, "target_normalization_mode", "none")
    if mode == "global_per_coordinate":
        mean, std = dataset.data_loader.load_behaviour_global_stats(behaviour)
        coord_dim = mean.shape[-1]
        reshaped = windows.reshape(len(windows), -1, coord_dim)
        normalized = (reshaped - mean.reshape(1, 1, coord_dim)) / std.reshape(1, 1, coord_dim)
        return normalized.reshape(len(windows), -1).astype(np.float32)
    if mode in {"none", "per_trial"}:
        return windows
    raise ValueError(f"Unsupported target_normalization_mode for model input: {mode}")


def build_counterfactual_trial(
    model,
    dataset,
    source_idx,
    target_idx,
    behaviours,
    device,
    input_window,
    output_window,
    unknown_input_window=None,
    spike_target_mode=None,
    behaviour_target_mode=None,
    generated_decode_features="rates",
    generated_seed=0,
):
    source_inputs, source_behaviour = build_trial_windows(
        dataset,
        source_idx,
        behaviours,
        input_window,
        output_window,
        unknown_input_window=unknown_input_window,
        spike_target_mode=spike_target_mode,
        behaviour_target_mode=behaviour_target_mode,
    )
    target_inputs, target_behaviour = build_trial_windows(
        dataset,
        target_idx,
        behaviours,
        input_window,
        output_window,
        unknown_input_window=unknown_input_window,
        spike_target_mode=spike_target_mode,
        behaviour_target_mode=behaviour_target_mode,
    )

    source_batch = {key: torch.from_numpy(value).to(device) for key, value in source_inputs.items()}
    cf_batch = {key: torch.from_numpy(value).to(device) for key, value in source_inputs.items()}
    for behaviour in behaviours:
        source_model_behaviour = prepare_behaviour_for_model_input(
            dataset, behaviour, source_behaviour[behaviour]
        )
        source_batch[behaviour] = torch.from_numpy(source_model_behaviour).to(device)
        target_model_behaviour = prepare_behaviour_for_model_input(
            dataset, behaviour, target_behaviour[behaviour]
        )
        cf_batch[behaviour] = torch.from_numpy(target_model_behaviour).to(device)

    with torch.no_grad():
        source_recon = model(source_batch)["reconstruction"].squeeze(-1).detach().cpu().numpy()
        cf_rates = model(cf_batch)["reconstruction"].squeeze(-1).detach().cpu().numpy()
    cf_activity = generated_activity_to_decoder_features(
        cf_rates,
        mode=generated_decode_features,
        seed=generated_seed,
    )

    return {
        "source_spikes": source_inputs["spikes"],
        "source_reconstruction": source_recon,
        "target_spikes": target_inputs["spikes"],
        "source_behaviour": source_behaviour,
        "target_behaviour": target_behaviour,
        "counterfactual_rates": cf_rates,
        "counterfactual_spikes": cf_activity,
    }


def _prepare_train_behaviours(train_dataset, behaviours, input_window, output_window, target_mode):
    return {
        behaviour: window_condition_target(
            train_dataset.data_loader.load_behaviour(behaviour),
            input_window=input_window,
            output_window=output_window,
            delay_factor=0,
            target_mode=target_mode,
        )
        for behaviour in behaviours
    }


def fit_shared_decoders(
    train_dataset,
    behaviours,
    input_window,
    output_window,
    decoder_type="cebra",
    hidden_dim=64,
    decoder_kwargs=None,
    spike_target_mode=None,
    behaviour_target_mode=None,
):
    train_spikes = train_dataset.spikes.data.numpy()
    behaviour_mode = behaviour_target_mode or spike_target_mode or "future"
    train_behaviours = _prepare_train_behaviours(
        train_dataset,
        behaviours,
        input_window,
        output_window,
        behaviour_mode,
    )
    return {
        behaviour: fit_decoder(
            train_spikes,
            train_behaviours[behaviour],
            decoder_type=decoder_type,
            hidden_dim=hidden_dim,
            decoder_kwargs=decoder_kwargs,
        )
        for behaviour in behaviours
    }


def compute_pair_metrics(
    module,
    train_dataset,
    val_dataset,
    behaviours,
    device,
    input_window,
    output_window,
    unknown_input_window=None,
    spike_target_mode=None,
    behaviour_target_mode=None,
    decoder_type="cebra",
    hidden_dim=64,
    decoder_kwargs=None,
    decoders=None,
    generated_decode_features="rates",
    generated_seed=0,
):
    if decoders is None:
        decoders = fit_shared_decoders(
            train_dataset,
            behaviours,
            input_window,
            output_window,
            decoder_type=decoder_type,
            hidden_dim=hidden_dim,
            decoder_kwargs=decoder_kwargs,
            spike_target_mode=spike_target_mode,
            behaviour_target_mode=behaviour_target_mode,
        )

    trial_info = val_dataset.data_loader.load_trial_info().reset_index(drop=True)
    trial_pairs = build_relaxed_pairs(trial_info)
    pair_batches = []
    per_pair = []
    for source_idx, target_idx in trial_pairs:
        batch = build_counterfactual_trial(
            module.model,
            val_dataset,
            source_idx,
            target_idx,
            behaviours,
            device,
            input_window,
            output_window,
            unknown_input_window=unknown_input_window,
            spike_target_mode=spike_target_mode,
            behaviour_target_mode=behaviour_target_mode,
            generated_decode_features=generated_decode_features,
            generated_seed=generated_seed + int(source_idx) * 1000003 + int(target_idx),
        )
        pair_batches.append(batch)
        per_pair.append(
            {
                "source_idx": int(source_idx),
                "target_idx": int(target_idx),
                "pairing_strategy": PAIRING_STRATEGY,
                "behaviours": {},
                "arrays": {},
                "spikes": {
                    "source": batch["source_spikes"],
                    "target": batch["target_spikes"],
                    "counterfactual": batch["counterfactual_spikes"],
                },
                "rates": {
                    "source_reconstruction": batch["source_reconstruction"],
                    "counterfactual": batch["counterfactual_rates"],
                },
            }
        )

    for behaviour in behaviours:
        decoder = decoders[behaviour]
        source_true_batches = [batch["source_behaviour"][behaviour] for batch in pair_batches]
        target_true_batches = [batch["target_behaviour"][behaviour] for batch in pair_batches]
        source_spike_batches = [batch["source_spikes"] for batch in pair_batches]
        target_spike_batches = [batch["target_spikes"] for batch in pair_batches]
        cf_spike_batches = [batch["counterfactual_spikes"] for batch in pair_batches]

        source_decoded_batches = batched_decoder_predict(decoder, source_spike_batches)
        target_decoded_batches = batched_decoder_predict(decoder, target_spike_batches)
        cf_decoded_batches = batched_decoder_predict(decoder, cf_spike_batches)

        for idx, pair in enumerate(per_pair):
            source_true = source_true_batches[idx]
            target_true = target_true_batches[idx]
            source_decoded = source_decoded_batches[idx]
            target_decoded = target_decoded_batches[idx]
            cf_decoded = cf_decoded_batches[idx]
            cf_vs_source = float(r2_score(source_true, cf_decoded))
            cf_vs_target = float(r2_score(target_true, cf_decoded))
            target_real = float(r2_score(target_true, target_decoded))
            delta = cf_vs_target - cf_vs_source
            pair["behaviours"][behaviour] = {
                "decoder_r2_on_target_real": target_real,
                "counterfactual_decoded_vs_source_r2": cf_vs_source,
                "counterfactual_decoded_vs_target_r2": cf_vs_target,
                "target_minus_source": float(delta),
            }
            pair["arrays"][behaviour] = {
                "source_true": source_true,
                "target_true": target_true,
                "source_decoded": source_decoded,
                "target_decoded": target_decoded,
                "cf_decoded": cf_decoded,
            }

    for pair in per_pair:
        deltas = [pair["behaviours"][behaviour]["target_minus_source"] for behaviour in behaviours]
        cf_target_scores = [
            pair["behaviours"][behaviour]["counterfactual_decoded_vs_target_r2"] for behaviour in behaviours
        ]
        target_real_scores = [pair["behaviours"][behaviour]["decoder_r2_on_target_real"] for behaviour in behaviours]
        pair["mean_target_minus_source"] = float(np.mean(deltas))
        pair["mean_counterfactual_vs_target_r2"] = float(np.mean(cf_target_scores))
        pair["mean_target_real_r2"] = float(np.mean(target_real_scores))

    for row in per_pair:
        row["trial_info_source"] = trial_info.iloc[row["source_idx"]].to_dict()
        row["trial_info_target"] = trial_info.iloc[row["target_idx"]].to_dict()
    return per_pair, decoders


def _cv_folds(n, cv_folds, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [fold for fold in np.array_split(perm, cv_folds) if len(fold) > 0]


def _ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1))


def _normalize_between_controls(score, source_control, target_control):
    score = np.asarray(score, dtype=float)
    source_control = np.asarray(source_control, dtype=float)
    target_control = np.asarray(target_control, dtype=float)
    denom = target_control - source_control
    normalized = np.full_like(score, np.nan, dtype=float)
    valid = np.abs(denom) > 1e-8
    normalized[valid] = (score[valid] - source_control[valid]) / denom[valid]
    if normalized.ndim == 0:
        return float(normalized)
    return normalized
