import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors
from pathlib import Path
import os
from .data_helpers import DataHelpers

import warnings
import pandas as pd
import re


NLB_CONDITION_FIELDS = {
    "mc_maze": ["trial_type", "trial_version"],
    "mc_maze_large": ["trial_type", "trial_version"],
    "mc_maze_medium": ["trial_type", "trial_version"],
    "mc_maze_small": ["trial_type", "trial_version"],
    "area2_bump": ["cond_dir", "ctr_hold_bump"],
    "dmfc_rsg": ["is_eye", "theta", "is_short", "ts"],
}


def _inject_synthetic_neurons(data, num_synth_neurons):
    if num_synth_neurons <= 0:
        return data

    mean_rate = float(np.mean(data))
    synth_shape = (data.shape[0], data.shape[1], num_synth_neurons)
    synth_spikes = np.random.poisson(mean_rate, size=synth_shape).astype(data.dtype, copy=False)
    return np.concatenate((data, synth_spikes), axis=2)


class SpikingData:
    def __init__(self, data, input_window, output_window, delay_factor, normalize, target_mode="future"):
        total_window = _total_window(input_window, output_window, delay_factor, target_mode)

        if normalize:
            data = DataHelpers.normalize(data)

        time_points = []
        for trial in data:
            n_time = trial.shape[0]
            for t in range(n_time - total_window + 1):
                neural_activity_window = trial[
                    t : t + input_window
                ]  # (input_window, num_neurons)
                neural_activity_window = np.expand_dims(
                    np.transpose(neural_activity_window), axis=0
                )
                time_points.append(neural_activity_window)
        data = np.vstack(time_points)
        # data is shape (num_timepoints, num_neurons, input_window)
        data = torch.from_numpy(data).float()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class ContextSpikingData:
    def __init__(self, data, input_window, output_window, delay_factor, context_window, normalize, target_mode="future"):
        total_window = _total_window(input_window, output_window, delay_factor, target_mode)

        if normalize:
            data = DataHelpers.normalize(data)

        time_points = []
        for trial in data:
            n_time = trial.shape[0]
            for t in range(n_time - total_window + 1):
                input_end = t + input_window
                context_start = max(0, input_end - context_window)
                context_window_data = np.zeros((context_window, trial.shape[1]), dtype=trial.dtype)
                valid = trial[context_start:input_end]
                context_window_data[-len(valid) :] = valid
                context_window_data = np.expand_dims(
                    np.transpose(context_window_data), axis=0
                )
                time_points.append(context_window_data)
        data = np.vstack(time_points)
        data = torch.from_numpy(data).float()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def _total_window(input_window, output_window, delay_factor, target_mode):
    if target_mode == "future":
        return input_window + output_window + delay_factor
    if target_mode == "input_trailing":
        return input_window
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def _target_bounds(t, input_window, output_window, delay_factor, target_mode):
    if target_mode == "future":
        out_start = t + input_window + delay_factor
        out_end = out_start + output_window
        return out_start, out_end
    if target_mode == "input_trailing":
        out_end = t + input_window
        out_start = out_end - output_window
        return out_start, out_end
    raise ValueError(f"Unsupported target_mode: {target_mode}")


class SpikeTargetData:
    def __init__(self, data, input_window, output_window, delay_factor, normalize, target_mode="future"):
        if normalize:
            data = DataHelpers.normalize(data)

        total_window = _total_window(input_window, output_window, delay_factor, target_mode)
        time_points = []
        for trial in data:
            n_time = trial.shape[0]
            for t in range(n_time - total_window + 1):
                out_start, out_end = _target_bounds(
                    t, input_window, output_window, delay_factor, target_mode
                )
                neural_target_window = trial[out_start:out_end]
                neural_target_window = np.expand_dims(
                    np.transpose(neural_target_window), axis=0
                )
                time_points.append(neural_target_window)
        data = np.vstack(time_points)
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class DataTarget:
    def __init__(self, data, id, input_window, output_window, delay_factor, normalize, target_mode="future"):
        if normalize:
            data = DataHelpers.normalize(data)

        total_window = _total_window(input_window, output_window, delay_factor, target_mode)
        time_points = []
        for trial in data:
            n_time = trial.shape[0]
            for t in range(n_time - total_window + 1):
                out_start, out_end = _target_bounds(
                    t, input_window, output_window, delay_factor, target_mode
                )
                target_window = trial[out_start:out_end]
                time_points.append(target_window.flatten())

        self.data = torch.from_numpy(np.vstack(time_points)).float()

        self.id = id

    def get_id(self):
        return self.id

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]



class NLBDatasetLoader:
    def __init__(
        self,
        dataset_name,
        flag,
        data_filepath,
        smoothing_constant,
        bin_width,
    ):
        self.dataset_loaded = False
        self.dataset_name = dataset_name
        self.smoothing_constant = smoothing_constant
        self.bin_width = bin_width
        self.flag = flag
        self.data_filepath = data_filepath

        self.loaded_dataset_dir = f"./loaded_datasets/{dataset_name}/smooth={smoothing_constant}_bin={bin_width}/{flag}"
        self.shared_dataset_dir = f"./loaded_datasets/{dataset_name}/smooth={smoothing_constant}_bin={bin_width}"

        os.makedirs(self.loaded_dataset_dir, exist_ok=True)
        os.makedirs(self.shared_dataset_dir, exist_ok=True)

    def load_behaviour_global_stats(self, name):
        stats_path = Path(f"{self.shared_dataset_dir}/target_stats_{name}.npz")
        if stats_path.is_file():
            stats = np.load(stats_path)
            return stats["mean"], stats["std"]

        train_loader = NLBDatasetLoader(
            self.dataset_name,
            "train",
            self.data_filepath,
            self.smoothing_constant,
            self.bin_width,
        )
        train_behaviour = train_loader.load_behaviour(name).astype(np.float32)
        mean = train_behaviour.mean(axis=(0, 1), keepdims=True)
        std = train_behaviour.std(axis=(0, 1), keepdims=True) + 1e-8
        np.savez(stats_path, mean=mean, std=std)
        return mean, std

    def _load_trial_info_behavior(self, name):
        trial_info = self.load_trial_info()
        tlen = self.load_spikes().shape[1]

        if name == "cond_dir_sincos":
            angles = trial_info["cond_dir"].to_numpy(dtype=np.float32)
            max_abs = np.nanmax(np.abs(angles)) if len(angles) else 0.0
            if max_abs > (2 * np.pi + 1e-3):
                angles = np.deg2rad(angles)
            values = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        elif name == "target_dir_sincos":
            def parse_target_pos(s):
                nums = [int(x) for x in re.findall(r"-?\d+", str(s))]
                if len(nums) == 0 or len(nums) % 2 != 0:
                    return None
                return np.array(nums, dtype=float).reshape(-1, 2)

            def active_target_xy(row):
                arr = parse_target_pos(row["target_pos"])
                if arr is None:
                    return np.array([np.nan, np.nan], dtype=float)
                idx = int(row["active_target"]) if not pd.isna(row["active_target"]) else 0
                idx = max(0, min(idx, len(arr) - 1))
                return arr[idx]

            xy = np.vstack([active_target_xy(row) for _, row in trial_info.iterrows()])
            angles = (np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) + 360.0) % 360.0
            angles = np.deg2rad(angles.astype(np.float32))
            values = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        else:
            values = (
                trial_info[name]
                .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
                .to_numpy()
                .astype(np.float32)
            )
            if values.ndim == 1:
                values = values[:, None]

        return np.repeat(values[:, None, :], tlen, axis=1)

    def _condition_fields(self):
        if self.dataset_name not in NLB_CONDITION_FIELDS:
            raise ValueError(
                f"Condition-based utilities are not configured for dataset {self.dataset_name}"
            )
        return NLB_CONDITION_FIELDS[self.dataset_name]

    def _condition_list(self, trial_info):
        fields = self._condition_fields()
        unique_conditions = (
            trial_info[fields]
            .dropna()
            .drop_duplicates()
            .sort_values(fields)
        )
        return unique_conditions.reset_index(drop=True)

    def _condition_mask(self, trial_info, condition_idx):
        fields = self._condition_fields()
        conditions = self._condition_list(trial_info)
        target_condition = conditions.iloc[condition_idx]
        return np.all(trial_info[fields].to_numpy() == target_condition.to_numpy(), axis=1)

    def load_dataset(self):
        self.dataset = NWBDataset(self.data_filepath)
        if self.smoothing_constant > 0:
            self.dataset.smooth_spk(gauss_width=self.smoothing_constant, overwrite=True)

        if self.bin_width > 1:
            self.dataset.resample(self.bin_width)

        self.dataset_loaded = True

    def load_behaviour(self, name):
        file_path = f"{self.loaded_dataset_dir}/{name}.h5"

        if Path(file_path).is_file():
            with h5py.File(file_path, "r") as behaviour_tensor:
                if "train_behavior" in behaviour_tensor:
                    return behaviour_tensor["train_behavior"][:]
                if "data" in behaviour_tensor:
                    return behaviour_tensor["data"][:]
                raise KeyError(
                    f"Behaviour cache {file_path} does not contain 'train_behavior' or 'data'"
                )
        elif name in {"cond_dir_sincos", "target_dir_sincos"} or name in self.load_trial_info().columns:
            behaviour = self._load_trial_info_behavior(name)
            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=behaviour)
            return behaviour
        else:
            if self.dataset_loaded is False:
                self.load_dataset()

            update_params = {"behavior_field": name}
            behaviour_tensor = make_train_input_tensors(
                self.dataset,
                dataset_name=self.dataset_name,
                update_params=update_params,
                save_file=True,
                save_path=file_path,
                include_behavior=True,
                trial_split=self.flag,
            )

        return behaviour_tensor["train_behavior"][:]

    def load_behaviour_condition(self, name, condition_idx):
        behaviour = self.load_behaviour(name)
        trial_info = self.load_trial_info()
        condition_mask = self._condition_mask(trial_info, condition_idx)
        return behaviour[condition_mask]

    def load_behaviour_condition_avg(self, name, condition_idx):
        return np.mean(self.load_behaviour_condition(name, condition_idx), axis=0)

    def load_spikes(self):
        file_path = f"{self.loaded_dataset_dir}/spikes.h5"

        if Path(file_path).is_file():
            spikes_tensor = h5py.File(file_path, "r")
        else:
            if self.dataset_loaded is False:
                self.load_dataset()

            spikes_tensor = make_train_input_tensors(
                self.dataset,
                dataset_name=self.dataset_name,
                save_file=True,
                save_path=file_path,
                include_behavior=False,
                trial_split=self.flag,
            )
        return spikes_tensor["train_spikes_heldin"][:]

    def load_spikes_for_condition(self, condition_idx):
        """Load all spikes (trials) for a specific condition

        Returns:
            cond_spikes: array of (num_trials_condition, num_timepoints, num_neurons)
        """
        spikes = self.load_spikes()
        trial_info = self.load_trial_info()
        condition_mask = self._condition_mask(trial_info, condition_idx)
        return spikes[condition_mask]


    def load_data_for_condition(self, condition_idx, behaviours):
        """Load all data (spikes and target) for a specific condition

        Returns:
            data: list of trial in that condition, which each trial being a dict with 'spikes' and target keys, with corresponding tensors
        """
        condition_dict = {}

        condition_dict['spikes'] = self.load_spikes_for_condition(condition_idx)
        for behaviour in behaviours:
          condition_dict[behaviour] = self.load_behaviour_condition(behaviour, condition_idx)

        return condition_dict

    def load_spikes_condition_avg(self, condition_idx):
        return np.mean(self.load_spikes_for_condition(condition_idx), axis=0)


    def load_condition_averaged_data(self, behaviours):
        """Load condition-averaged spikes and behaviour targets.

        Returns:
            cond_avg_spikes: concatenation of condition-averaged spike arrays
            cond_avg_targets: dict mapping behaviour name -> concatenation of condition-averaged behaviour arrays
        """
        num_conditions = len(self._condition_list(self.load_trial_info()))
        cond_avg_spikes = []
        cond_avg_targets = {b: [] for b in behaviours}

        for cond_idx in range(num_conditions):
            cond_avg_spikes.append(self.load_spikes_condition_avg(cond_idx))
            for behaviour in behaviours:
                cond_avg_targets[behaviour].append(
                    self.load_behaviour_condition_avg(behaviour, cond_idx)
                )

        cond_avg_spikes = np.vstack(cond_avg_spikes)

        for behaviour in behaviours:
          cond_avg_targets[behaviour] = np.vstack(cond_avg_targets[behaviour])

        return cond_avg_spikes, cond_avg_targets

    def load_trial_info(self):
      file_path = f"{self.loaded_dataset_dir}/trial_info.csv"

      if Path(file_path).is_file():
          trial_info = pd.read_csv(file_path)
      else:
        if self.dataset_loaded is False:
          self.dataset = NWBDataset(self.data_filepath)

        trial_info = self.dataset.trial_info
        trial_info.to_csv(file_path)

      return trial_info[trial_info['split'] == self.flag]

class NLBDataset(Dataset):
    def __init__(
        self,
        data_filepath,
        flag,
        input_window,
        output_window,
        dataset_name,
        bin_width,
        smoothing_constant,
        normalize_spikes,
        normalize_targets,
        target_normalization_mode,
        delay_factor,
        behaviours=None,
        unknown_input_window=None,
        spike_target_mode=None,
        behaviour_target_mode=None,
        num_synth_neurons=0,
    ):
        if behaviours is None:
            target_names = ["eye_pos", "hand_pos", "hand_vel"]
        else:
            target_names = behaviours

        if delay_factor != 0:
            warnings.warn(
                "Delay factor already implemented in NLB datasets. Ignoring delay in config."
            )

        delay_factor = 0

        self.targets = []
        self.input_window = input_window
        self.output_window = output_window
        self.spike_target_mode = spike_target_mode
        self.behaviour_target_mode = behaviour_target_mode or spike_target_mode or "future"
        if target_normalization_mode is None:
            self.target_normalization_mode = "per_trial" if normalize_targets else "none"
        else:
            self.target_normalization_mode = target_normalization_mode

        if flag == 'test':
          nlb_flag = 'val'
          # test data is not public for NLB so we use val as test and part of train as val.
        elif flag == 'train':
          nlb_flag = 'train'
        else:
          raise ValueError("Val dataset not supported for NLB. Use test, or use validation dataloader from datamodule.")

        dataset_loader = NLBDatasetLoader(
            dataset_name,
            nlb_flag,
            data_filepath,
            smoothing_constant,
            bin_width,
        )  # ALL BEHAVIOURS: hand_pos, eye_pos, cursor_pos, hand_vel

        self.data_loader = dataset_loader
        raw_spikes = dataset_loader.load_spikes()
        raw_spikes = _inject_synthetic_neurons(raw_spikes, num_synth_neurons)
        self.spikes_input = None
        if spike_target_mode is None:
            self.spikes = SpikingData(
                raw_spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=delay_factor,
                normalize=normalize_spikes,
                target_mode="future",
            )
        else:
            self.spikes_input = SpikingData(
                raw_spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=delay_factor,
                normalize=normalize_spikes,
                target_mode=spike_target_mode,
            )
            self.spikes = SpikeTargetData(
                raw_spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=delay_factor,
                normalize=normalize_spikes,
                target_mode=spike_target_mode,
            )
        self.spikes_context = None
        if unknown_input_window is not None and unknown_input_window > input_window:
            self.spikes_context = ContextSpikingData(
                raw_spikes,
                input_window=input_window,
                output_window=output_window,
                delay_factor=delay_factor,
                context_window=unknown_input_window,
                normalize=normalize_spikes,
                target_mode=(spike_target_mode or "future"),
            )

        for target in target_names:
            target_temp = dataset_loader.load_behaviour(target)
            target_normalize = False
            if self.target_normalization_mode == "global_per_coordinate":
                mean, std = dataset_loader.load_behaviour_global_stats(target)
                target_temp = (target_temp - mean) / std
            elif self.target_normalization_mode == "per_trial":
                target_normalize = True
            elif self.target_normalization_mode != "none":
                raise ValueError(
                    f"Unsupported target_normalization_mode: {self.target_normalization_mode}"
                )
            self.targets.append(
                DataTarget(
                    target_temp,
                    target,
                    input_window,
                    output_window,
                    delay_factor,
                    target_normalize,
                    target_mode=self.behaviour_target_mode,
                )
            )

    def __len__(self):
        return len(self.spikes)

    def get_group_ids(self):
        """Group flattened NLB windows by original trial for CV splits."""
        raw_spikes = self.data_loader.load_spikes()
        n_trials = int(raw_spikes.shape[0])
        if n_trials <= 0:
            raise ValueError("Cannot build NLB trial groups for an empty dataset.")

        n_samples = len(self)
        if n_samples % n_trials != 0:
            raise ValueError(
                f"NLB dataset has {n_samples} flattened samples for {n_trials} trials; "
                "cannot construct trial-level CV groups without partial trials."
            )

        windows_per_trial = n_samples // n_trials
        return np.repeat(np.arange(n_trials, dtype=np.int64), windows_per_trial)

    def __getitem__(self, idx):
        item = {}
        item["spikes"] = self.spikes[idx]
        if self.spikes_input is not None:
            item["spikes_input"] = self.spikes_input[idx]
        if self.spikes_context is not None:
            item["spikes_context"] = self.spikes_context[idx]

        for target in self.targets:
            item[target.get_id()] = target[idx]

        item['sample_index'] = idx

        return item
