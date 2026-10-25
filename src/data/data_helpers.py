from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch



class DataHelpers:
    def bin_spikes(spikes, bin_size):
        n_trials, T, n_neurons = spikes.shape

        T_trim = T - (T % bin_size)
        spikes_trim = spikes[:, :T_trim, :]

        spikes_binned = spikes_trim.reshape(
            n_trials, T_trim // bin_size, bin_size, n_neurons
        ).sum(axis=2)
        return spikes_binned / bin_size

    def normalize(dataset):
        mean = dataset.mean(axis=(1, 2), keepdims=True)
        std = (
            np.nanstd(dataset, axis=(1, 2), keepdims=True) + 1e-8
        )  # epsilon avoids division by zero
        dataset -= mean
        dataset /= std
        return dataset

    def normalize_2d(dataset):
        mean = dataset.nanmean(axis=0, keepdims=True)
        std = np.nanstd(dataset, axis=0, keepdims=True) + 1e-8

        dataset = (dataset - mean) / std
        return dataset

    def normalize_2d_bounded(tensor):
        min_val, max_val = tensor.min(0, keepdim=True)[0], tensor.max(0, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def gaussian_smoothing_2d(dataset, sigma=5.0):
        dataset_f32 = dataset.astype(np.float32)
        smoothed_dataset = gaussian_filter1d(dataset_f32, sigma=sigma, axis=0)
        return smoothed_dataset.astype(dataset.dtype)

    def gaussian_smoothing(dataset, sigma=5.0):
        dataset_f32 = dataset.astype(np.float32)
        dataset_f32 = gaussian_filter1d(dataset_f32, sigma=sigma, axis=1)
        smoothed_dataset = dataset_f32.astype(dataset.dtype)

        return smoothed_dataset


    def bin_data(behaviour_extracted, num_bins=11):
        x = behaviour_extracted[:, 0]
        x_bins = torch.bucketize(x, torch.linspace(x.min(), x.max(), num_bins))
        if behaviour_extracted.shape[1] == 1:
            return x_bins

        y = behaviour_extracted[:, 1]
        y_bins = torch.bucketize(y, torch.linspace(y.min(), y.max(), num_bins))
        final_bins = x_bins * (y_bins.max().item() + 1) + y_bins
        return final_bins

    def find_test_pairs(all_data, limit_ratio=0.1):
        pairs = []
        n_rows = len(all_data)
        test = []
        temp = []
        test_indices = []

        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                diff_count = (all_data[i] != all_data[j]).sum()
                if diff_count == 1:
                    pairs.append((i, j))
                    temp.append(all_data[i])
                    test.append(all_data[j])
                    test_indices.append(j)
                    break
            if len(test_indices) > limit_ratio * n_rows:
                break

        return pairs, temp, test, test_indices

    def remove_b_from_a(all_data, test, indices):
        diff = (all_data.unsqueeze(1) == test.unsqueeze(0))
        row_matches = diff.all(dim=2)
        is_in_test = row_matches.any(dim=1)
        train = all_data[~is_in_test.flatten()]
        train_indices = indices[~is_in_test.flatten()]
        return train, train_indices


    def find_mapping(train, test):
        train_map_index = []

        for i in range(len(train)):
            found_match = False
            for j in range(len(test)):
                diff_count = (train[i] != test[j]).sum()

                if diff_count == 1:
                    train_map_index.append(j)
                    found_match = True
                    break

            if not found_match:
                train_map_index.append(-1)

        return train_map_index
