import os
import sys

# Import this in scripts to access self-made packages (ie. src.utils)
current_dir = os.getcwd()
sys.path.append(current_dir)

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from .data_helpers import DataHelpers

import warnings
import pandas as pd


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


def _window_continuous_input(array, input_window, output_window, delay_factor, target_mode):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    n_time = array.shape[0]
    for t in range(n_time - total_window + 1):
        samples.append(array[t : t + input_window].T)
    if not samples:
        raise ValueError("No valid spike windows constructed for RatInABox dataset")
    return np.stack(samples, axis=0).astype(np.float32)


def _window_continuous_context(array, input_window, output_window, delay_factor, context_window, target_mode):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    n_time = array.shape[0]
    for t in range(n_time - total_window + 1):
        input_end = t + input_window
        context_start = max(0, input_end - context_window)
        context = np.zeros((context_window, array.shape[1]), dtype=array.dtype)
        valid = array[context_start:input_end]
        context[-len(valid) :] = valid
        samples.append(context.T)
    if not samples:
        raise ValueError("No valid spike context windows constructed for RatInABox dataset")
    return np.stack(samples, axis=0).astype(np.float32)


def _window_continuous_target(array, input_window, output_window, delay_factor, target_mode):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    n_time = array.shape[0]
    for t in range(n_time - total_window + 1):
        out_start, out_end = _target_bounds(
            t, input_window, output_window, delay_factor, target_mode
        )
        samples.append(array[out_start:out_end].reshape(-1))
    if not samples:
        raise ValueError("No valid behaviour windows constructed for RatInABox dataset")
    return np.stack(samples, axis=0).astype(np.float32)


def _window_continuous_spike_target(array, input_window, output_window, delay_factor, target_mode):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    n_time = array.shape[0]
    for t in range(n_time - total_window + 1):
        out_start, out_end = _target_bounds(
            t, input_window, output_window, delay_factor, target_mode
        )
        samples.append(array[out_start:out_end].T)
    if not samples:
        raise ValueError("No valid spike targets constructed for RatInABox dataset")
    return np.stack(samples, axis=0).astype(np.float32)


class RatInBoxDatasetLoader:
    def __init__(
        self,
        dataset_name,
        flag,
        data_filepath,
        smoothing_constant,
        bin_width,
        noise,
        num_cells,
        minutes,
        num_bins,
        use_counterfactual,
        manual_trajectory,
        include_noise_cell,
    ):
        self.dataset_loaded = False
        self.dataset_name = dataset_name
        self.smoothing_constant = smoothing_constant
        self.bin_width = bin_width
        self.flag = flag
        self.data_filepath = data_filepath
        self.noise = noise
        self.num_cells = num_cells
        self.minutes = minutes
        self.num_bins = num_bins
        self.use_counterfactual = use_counterfactual
        self.manual_trajectory = manual_trajectory
        self.include_noise_cell = include_noise_cell

        self.loaded_dataset_dir = f"./loaded_datasets/{dataset_name}/noise_cell={include_noise_cell}_cfact={use_counterfactual}_noise={noise}_num_cells={num_cells}_minutes={minutes}/{flag}"

        os.makedirs(self.loaded_dataset_dir, exist_ok=True)

    def load_dataset(self):
        dataset_name = f"custom_rat_in_a_box_dataset_type:_(train)_noise_cell:_{self.include_noise_cell}_noise:_{self.noise}_minutes:_{self.minutes}_num_cells:_{self.num_cells}.pkl"

        if self.flag == "shift_left" or self.flag == "shift_right":
            if self.flag == "shift_left":
                file = self.data_filepath + f"custom_rat_in_a_box_dataset_type:_(shifted)_noise_cell:_{self.include_noise_cell}_noise:_{self.noise}_minutes:_{self.minutes}_num_cells:_{self.num_cells}.pkl"
                self.dataset = pd.read_pickle(file)

            elif self.flag == "shift_right":
                file = self.data_filepath + f"custom_rat_in_a_box_dataset_type:_(cfact_shifted)_noise_cell:_{self.include_noise_cell}_noise:_{self.noise}_minutes:_{self.minutes}_num_cells:_{self.num_cells}.pkl"
                self.dataset = pd.read_pickle(file)

        else:
            file = self.data_filepath + dataset_name
            self.dataset = pd.read_pickle(file)

            if self.manual_trajectory:
                if self.flag == "test":
                    file = self.data_filepath + f"custom_rat_in_a_box_dataset_type:_(inverted)_noise_cell:_{self.include_noise_cell}_noise:_{self.noise}_minutes:_{self.minutes}_num_cells:_{self.num_cells}.pkl"
                    print("Loading manual-trajectory test split from inverted dataset")
                    dataset = pd.read_pickle(file)
                    split_idx = round(len(dataset) * 0.2)
                    self.dataset = dataset.iloc[:split_idx]
                self.use_counterfactual = False


            if self.use_counterfactual:
                train, test, train_mappings, train_indices, test_indices = self.create_counterfactual_dataset(self.dataset)

                if self.flag == "test":
                    self.dataset = self.dataset.iloc[test_indices]
                    self.dataset["map"] = np.full(len(self.dataset), -1, dtype=np.int32)
                    self.dataset["bin_indices"] = [slice for slice in test]
                else:
                    self.dataset = self.dataset.iloc[train_indices]
                    self.dataset["map"] = train_mappings
                    self.dataset["bin_indices"] = [slice for slice in train]



        self.dataset_loaded = True

    def create_counterfactual_dataset(self, dataset):


        binned_behaviours = []

        for behaviour in ["pos", "head_direction", "speed"]:

            if behaviour == "speed":
                tensor = np.linalg.norm(np.stack(dataset["vel"].values).astype(np.float32), axis=1)
            else:
                tensor = np.stack(dataset[behaviour].values).astype(np.float32)
            if tensor.ndim == 1:
                tensor = tensor[:, None]

            tensor = torch.from_numpy(tensor)
            tensor = DataHelpers.normalize_2d_bounded(tensor)
            labels = DataHelpers.bin_data(tensor, self.num_bins)
            binned_behaviours.append(labels.unsqueeze(1))

        all_binned = torch.stack(binned_behaviours, dim=1)

        limit_ratio = 0.1
        pairs, temp, test, test_indices = DataHelpers.find_test_pairs(all_binned, limit_ratio)


        temp = torch.stack(temp)
        test = torch.stack(test)

        train, train_indices = DataHelpers.remove_b_from_a(all_binned, test, np.array(range(len(all_binned))))


        train_mappings = DataHelpers.find_mapping(train, test)

        return train, test, train_mappings, train_indices, test_indices


    def load_behaviour(self, name):

        file_path = f"{self.loaded_dataset_dir}/{name}.h5"
        if Path(file_path).is_file():
            with h5py.File(file_path, "r") as f:
                behaviour_tensor = np.array(f["data"][:])
            if name == "speed" and behaviour_tensor.ndim == 2 and behaviour_tensor.shape[1] == 2:
                behaviour_tensor = behaviour_tensor[:, :1]
        else:
            if self.dataset_loaded is False:
                self.load_dataset()
            if name == "speed":

                behaviour_tensor = np.linalg.norm(np.stack(self.dataset["vel"].values).astype(np.float32), axis=1)
            else:
                behaviour_tensor = np.stack(self.dataset[name].values).astype(np.float32)

            if behaviour_tensor.ndim == 1:
                behaviour_tensor = behaviour_tensor[:, None]
            split_idx = round(len(behaviour_tensor) * 0.8)
            if self.flag == "train":
                behaviour_tensor = behaviour_tensor[0 : split_idx]
            elif self.flag == "val":
                behaviour_tensor = behaviour_tensor[split_idx : len(behaviour_tensor)]
            else:
                behaviour_tensor = behaviour_tensor



            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=behaviour_tensor)



        return behaviour_tensor

    def load_spikes(self):

        file_path = f"{self.loaded_dataset_dir}/spikes.h5"

        if Path(file_path).is_file():
            with h5py.File(file_path, "r") as f:
                spikes_tensor = np.array(f["data"][:])
        else:
            if self.dataset_loaded is False:
                self.load_dataset()


            spikes_tensor = np.stack(self.dataset["spikes"].values).astype(np.float32)

            split_idx = round(len(spikes_tensor) * 0.8)
            if self.flag == "train":
                spikes_tensor = spikes_tensor[0 : split_idx]
            elif self.flag == "val":
                spikes_tensor = spikes_tensor[split_idx : len(spikes_tensor)]
            else:
                spikes_tensor = spikes_tensor

            if self.smoothing_constant > 0:
                spikes_tensor = DataHelpers.gaussian_smoothing_2d(spikes_tensor, self.smoothing_constant)



            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=spikes_tensor)

        return spikes_tensor

    def load_rates(self):

        file_path = f"{self.loaded_dataset_dir}/rates.h5"
        if Path(file_path).is_file():
            with h5py.File(file_path, "r") as f:
                rates_tensor = np.array(f["data"][:])
        else:
            if self.dataset_loaded is False:
                self.load_dataset()


            rates_tensor = np.stack(self.dataset["rates"].values).astype(np.float32)

            split_idx = round(len(rates_tensor) * 0.8)
            if self.flag == "train":
                rates_tensor = rates_tensor[0 : split_idx]
            elif self.flag == "val":
                rates_tensor = rates_tensor[split_idx : len(rates_tensor)]

            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=rates_tensor)

        return rates_tensor



    def load_mappings(self):

        file_path = f"{self.loaded_dataset_dir}/map.h5"

        if Path(file_path).is_file():
            with h5py.File(file_path, "r") as f:
                map = np.array(f["data"][:])
        else:
            if self.dataset_loaded is False:
                self.load_dataset()


            map = np.stack(self.dataset["map"].values)

            split_idx = round(len(map) * 0.8)
            if self.flag == "train":
                map = map[0 : split_idx]
            elif self.flag == "val":
                map = map[split_idx : len(map)]
            else:
                map = map

            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=map)


        return map

    def load_bin_indices(self):

        file_path = f"{self.loaded_dataset_dir}/bins.h5"

        if Path(file_path).is_file():
            with h5py.File(file_path, "r") as f:
                bins = np.array(f["data"][:])
        else:
            if self.dataset_loaded is False:
                self.load_dataset()


            bins = np.stack(self.dataset["bin_indices"].values)

            split_idx = round(len(bins) * 0.8)
            if self.flag == "train":
                bins = bins[0 : split_idx]
            elif self.flag == "val":
                bins = bins[split_idx : len(bins)]
            else:
                bins = bins


            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=bins)


        return bins


class RatInBoxDatasetCustom(Dataset):
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
        delay_factor,
        noise,
        num_cells,
        minutes,
        num_bins,
        use_counterfactual,
        manual_trajectory,
        include_noise_cell,
        unknown_input_window=None,
        spike_target_mode=None,
        behaviour_target_mode=None,
    ):
        self.target_names = ["pos", "head_direction", "speed"]

        self.input_window = input_window
        self.output_window = output_window
        self.spike_target_mode = spike_target_mode
        self.behaviour_target_mode = behaviour_target_mode or spike_target_mode or "future"
        self.spikes_input = None
        self.spikes_context = None

        if delay_factor != 0 and spike_target_mode is None:
            warnings.warn(
                "Ignoring delay_factor because spike_target_mode is not set."
            )
            delay_factor = 0
        self.use_counterfactual = use_counterfactual
        self.manual_trajectory = manual_trajectory
        self.include_noise_cell = include_noise_cell

        self.targets = []

        dataset_loader = RatInBoxDatasetLoader(
            dataset_name,
            flag,
            data_filepath,
            smoothing_constant,
            bin_width,
            noise,
            num_cells,
            minutes,
            num_bins,
            self.use_counterfactual,
            self.manual_trajectory,
            self.include_noise_cell

        )

        data = dataset_loader.load_spikes().astype(np.float32)
        rates = dataset_loader.load_rates().astype(np.float32)
        if self.spike_target_mode is None:
            if self.input_window != 1 or self.output_window != 1:
                raise ValueError(
                    "RatInBoxDatasetCustom requires spike_target_mode when input_window or "
                    "output_window differs from 1."
                )
            self.data = torch.from_numpy(data).float()
            self.rates = torch.from_numpy(rates).float()
        else:
            if self.use_counterfactual:
                raise NotImplementedError(
                    "Windowed RatInABox targets are not implemented for loader-level "
                    "counterfactual pairing."
                )
            self.spikes_input = torch.from_numpy(
                _window_continuous_input(
                    data,
                    input_window=self.input_window,
                    output_window=self.output_window,
                    delay_factor=delay_factor,
                    target_mode=self.spike_target_mode,
                )
            ).float()
            if unknown_input_window is not None and unknown_input_window > self.input_window:
                self.spikes_context = torch.from_numpy(
                    _window_continuous_context(
                        data,
                        input_window=self.input_window,
                        output_window=self.output_window,
                        delay_factor=delay_factor,
                        context_window=unknown_input_window,
                        target_mode=self.spike_target_mode,
                    )
                ).float()
            self.data = torch.from_numpy(
                _window_continuous_spike_target(
                    data,
                    input_window=self.input_window,
                    output_window=self.output_window,
                    delay_factor=delay_factor,
                    target_mode=self.spike_target_mode,
                )
            ).float()
            self.rates = torch.from_numpy(
                _window_continuous_spike_target(
                    rates,
                    input_window=self.input_window,
                    output_window=self.output_window,
                    delay_factor=delay_factor,
                    target_mode=self.spike_target_mode,
                )
            ).float()

        for target in self.target_names:
            target_temp = dataset_loader.load_behaviour(target).astype(np.float32)
            if self.spike_target_mode is not None:
                target_temp = _window_continuous_target(
                    target_temp,
                    input_window=self.input_window,
                    output_window=self.output_window,
                    delay_factor=delay_factor,
                    target_mode=self.behaviour_target_mode,
                )
            self.targets.append(torch.from_numpy(target_temp).float())

        if self.use_counterfactual:
            self.mapping = torch.from_numpy(dataset_loader.load_mappings())
            self.bin_indices = torch.from_numpy(dataset_loader.load_bin_indices())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        # Reshape data from [num_neurons] to [num_neurons, input_window]
        data = self.data[idx]
        item["spikes"] = data.unsqueeze(-1) if data.dim() == 1 else data
        if self.spikes_input is not None:
            item["spikes_input"] = self.spikes_input[idx]
        if self.spikes_context is not None:
            item["spikes_context"] = self.spikes_context[idx]
        rates = self.rates[idx]
        item['rates'] =  rates.unsqueeze(-1) if rates.dim() == 1 else rates
        item['sample_index'] = idx

        for i, target in enumerate(self.targets):
            target_data = self.targets[i][idx]
            item[self.target_names[i]] = target_data

        if self.use_counterfactual:
            item["map"] = self.mapping[idx]
            item["bin_indices"] = self.bin_indices[idx]

        item['sample_index'] = idx

        return item
