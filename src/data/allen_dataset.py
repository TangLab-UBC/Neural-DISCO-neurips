import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os
from .data_helpers import DataHelpers

import pandas as pd


class AllenDatasetLoader:
    def __init__(
        self,
        dataset_name,
        flag,
        data_filepath,
        bin_width,
        session_id,
    ):

        self.dataset_loaded = False
        self.dataset_name = dataset_name
        self.bin_width = bin_width
        self.flag = flag
        self.data_filepath = data_filepath
        self.session_id = session_id
        self.has_pupil = True
        # 2 s presentation, so we get 2000 / self.bin_width bins per each presentation.
        self.bins_per_pres = 2000 / self.bin_width

        self.UNIT_METRICS = ['unit_ids', 'run_pval_dg', 'g_osi_dg', 'g_dsi_dg', 'run_mod_dg']

        loaded_dataset_dir = f"./loaded_datasets/{dataset_name}/sess{session_id}bin={bin_width}_presentation_aligned"
        os.makedirs(loaded_dataset_dir, exist_ok=True)
        self.dataset_fp = f"{loaded_dataset_dir}/{flag}.h5"

        if Path(self.dataset_fp).is_file():
          with h5py.File(self.dataset_fp, "r") as dataset:
            if 'pupil_size' not in dataset.keys():
              print("No pupil data for this session. Dataloader will not return pupil size.")
              self.has_pupil = False

    def load_and_bin_dataset(self):
        print(f'Loading and binning {self.flag} session {self.session_id}, this may take a few minutes.')
        self.load_session_metadata()
        epochs = self.session.get_stimulus_epochs()
        target_epochs = epochs[epochs['stimulus_name'] == 'drifting_gratings']
        if self.flag == 'train':
          target_epochs = target_epochs[:2]
        else:
          target_epochs = target_epochs[-1:]

        # Allen default unit-quality thresholds
        min_snr = 1
        max_isi = 0.5

        good_units_ids = self.session.units[
          (self.session.units.snr>min_snr)&(self.session.units.isi_violations<max_isi)
        ].index.values

        bin_width = self.bin_width / 1000 # ms -> seconds
        gaze_data = self.session.get_screen_gaze_data()
        if gaze_data is None:
          print("No pupil data for this session. Dataloader will not return pupil size.")
          self.has_pupil = False

        full_dataset_dict = {
          'speed': [],
          'condition' : [],
          'spikes': [],
          'orientation': [],
          'temporal_frequency': [],
          'unit_ids': [],
          'run_pval_dg': [],
          'g_osi_dg': [],
          'g_dsi_dg': [],
          'run_mod_dg': [],
        }

        if self.has_pupil is True:
          full_dataset_dict['pupil_size'] = []

        full_dataset_dict['presentation_id'] = []
        full_dataset_dict['presentation_bin'] = []

        bin_starts = []
        bin_ends = []
        presentations = self.session.stimulus_presentations
        for epoch in target_epochs.itertuples(index=False):
          epoch_presentations = presentations[
              (presentations['stimulus_name'] == 'drifting_gratings')
              & (presentations['start_time'] >= epoch.start_time)
              & (presentations['stop_time'] <= epoch.stop_time)
          ]
          for presentation_id, presentation in epoch_presentations.iterrows():
            edges = self.presentation_bin_edges(presentation, bin_width)
            if len(edges) <= 1:
              continue
            # n_bins will always be self.bins_per_pres
            n_bins = len(edges) - 1
            binned_condition_ids = np.full(
                n_bins,
                int(presentation.stimulus_condition_id),
                dtype=np.int64,
            )

            full_dataset_dict['speed'].append(self.bin_running_speed(edges))

            if self.has_pupil is True:
              full_dataset_dict['pupil_size'].append(self.bin_pupil_size(edges, gaze_data))

            full_dataset_dict['condition'].append(binned_condition_ids)
            full_dataset_dict['orientation'].append(self.get_condition_attr(binned_condition_ids, 'orientation'))
            full_dataset_dict['temporal_frequency'].append(self.get_condition_attr(binned_condition_ids, 'temporal_frequency'))
            full_dataset_dict['presentation_id'].append(np.full(n_bins, int(presentation_id), dtype=np.int64))
            full_dataset_dict['presentation_bin'].append(np.arange(n_bins, dtype=np.int64))
            bin_starts.append(edges[:-1])
            bin_ends.append(edges[1:])

        if not bin_starts:
          raise ValueError(
              f"No drifting-grating presentations found for {self.flag} "
              f"session {self.session_id}."
          )

        full_dataset_dict['spikes'] = [
            self.bin_spikes_for_intervals(
                good_units_ids,
                np.concatenate(bin_starts),
                np.concatenate(bin_ends),
            )
        ]


        # Wrap per-unit arrays in lists so the np.concatenate below accepts them

        full_dataset_dict['unit_ids'] = [good_units_ids]
        full_dataset_dict['run_pval_dg'] = [self.get_units_analysis_metric(good_units_ids, 'run_pval_dg')]
        full_dataset_dict['g_osi_dg'] = [self.get_units_analysis_metric(good_units_ids, 'g_osi_dg')]
        full_dataset_dict['g_dsi_dg'] = [self.get_units_analysis_metric(good_units_ids, 'g_dsi_dg')]
        full_dataset_dict['run_mod_dg'] = [self.get_units_analysis_metric(good_units_ids, 'run_mod_dg')]
        with h5py.File(self.dataset_fp, 'w') as hf:
          for key, val in full_dataset_dict.items():
              data = np.concatenate(val)
              hf.create_dataset(key, data=data)


    def get_num_good_units(self):
      self.load_session_metadata()
      # Allen default unit-quality thresholds
      min_snr = 1
      max_isi = 0.5
      num_units = np.count_nonzero((self.session.units.snr>min_snr)&(self.session.units.isi_violations<max_isi))
      return num_units

    def presentation_bin_edges(self, presentation, bin_width):
      n_bins = self.bins_per_pres
      # Instead of calculating the amount of bins for each presentation, we assume it to be always constant.
      # This prevents rare errors where some presentations are slightly shorter than 2 seconds, we just treat them all as 2 seconds.
      return presentation.start_time + np.arange(n_bins + 1) * bin_width

    def get_units_analysis_metric(self, unit_ids, metric):
      # requires session metadata saved previously (self.cache defined)
      good_unit_analysis_metrics = self.cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1').loc[unit_ids]
      results = good_unit_analysis_metrics[metric].values
      return results


    def get_condition_attr(self, binned_condition_ids, attr_name):
      # Map from condition ids to the attribute name.

      conds = self.session.stimulus_conditions
      binned_attribute_values = conds.iloc[binned_condition_ids][attr_name].values
      # Above maps null to str 'null'. Return as NaN
      binned_attribute_values = pd.to_numeric(binned_attribute_values, errors='coerce')
      return binned_attribute_values


    def bin_spikes_for_intervals(self, unit_ids, starts, ends):
      print(f"Binning spikes for {len(starts)} presentation-aligned bins and {len(unit_ids)} units.")
      spikes = np.zeros((len(starts), len(unit_ids)))
      for idx, unit_id in enumerate(unit_ids):
        spike_times = np.sort(self.session.spike_times[unit_id])
        left = np.searchsorted(spike_times, starts, side='left')
        right = np.searchsorted(spike_times, ends, side='left')
        spikes[:, idx] = right - left
      return spikes

    def bin_running_speed(self, edges):
      # Bin speed given by self.session.running_speed into bin edges
      running_speed = self.session.running_speed
      speed = self.bin_behaviour(running_speed.start_time.values, running_speed.velocity.values, edges)
      return speed

    def bin_pupil_size(self, edges, gaze_data):
      pupil_size = gaze_data['raw_eye_area']
      pupil_size_times = pupil_size.index.values
      pupil_size_vals = pupil_size.values
      pupil_size = self.bin_behaviour(pupil_size_times, pupil_size_vals, edges)
      return pupil_size

    def bin_behaviour(self, timestamps, behaviour, bin_edges):
      # Given bin edges and timestamped behavioural signal, take the mean
      # signal over all recorded timesteps inside each bin. Empty bins are
      # linearly interpolated.

      finite = np.isfinite(timestamps) & np.isfinite(behaviour)
      timestamps = timestamps[finite]
      behaviour = behaviour[finite]
      if len(timestamps) == 0:
        raise ValueError("Cannot bin behaviour with no finite samples.")

      in_range = (timestamps >= bin_edges[0]) & (timestamps < bin_edges[-1])
      local_timestamps = timestamps[in_range]
      local_behaviour = behaviour[in_range]
      bin_indices = np.digitize(local_timestamps, bin_edges) - 1  # make 0-indexed
      n_bins = len(bin_edges) - 1

      # Sum velocities per bin and count entries per bin
      attr_sum = np.bincount(bin_indices, weights=local_behaviour, minlength=n_bins)
      counts  = np.bincount(bin_indices, minlength=n_bins)

      # Mean velocity per bin (avoid divide-by-zero for empty bins)
      mean_attr = np.full(n_bins, np.nan, dtype=float)
      np.divide(attr_sum, counts, out=mean_attr, where=counts > 0)

      nans = np.isnan(mean_attr)
      if np.any(nans):
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mean_attr[nans] = np.interp(centers[nans], timestamps, behaviour)
      return mean_attr

    def load_session_metadata(self):
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

        manifest_path = os.path.join(self.data_filepath, "manifest.json")
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session = self.cache.get_session_data(self.session_id)

    def load_behaviour(self, name):
        if not Path(self.dataset_fp).is_file():
            self.load_and_bin_dataset()
        with h5py.File(self.dataset_fp, "r") as dataset:
          if name not in dataset.keys():
            raise UserWarning(f'No {name} key in dataset. Only {dataset.keys()}')
          return torch.from_numpy(dataset[name][:]).float()[:, None]

    def load_metadata(self, name):
        if not Path(self.dataset_fp).is_file():
            self.load_and_bin_dataset()
        with h5py.File(self.dataset_fp, "r") as dataset:
          if name not in dataset.keys():
            raise UserWarning(f'No {name} key in dataset. Only {dataset.keys()}')
          return torch.from_numpy(dataset[name][:]).long()

    def load_spikes(self):
        if not Path(self.dataset_fp).is_file():
            self.load_and_bin_dataset()
        with h5py.File(self.dataset_fp, "r") as dataset:
          return torch.from_numpy(dataset['spikes'][:]).float()[:, :, None]

    def load_unit_metrics(self):
      if not Path(self.dataset_fp).is_file():
            self.load_and_bin_dataset()

      with h5py.File(self.dataset_fp, "r") as dataset:
        return_dict = {}
        for metric in self.UNIT_METRICS:
          return_dict[metric] = torch.from_numpy(dataset[metric][:]).float()
      return return_dict

class AllenSpikes:
    def __init__(self, data, num_synth_neurons):
        if num_synth_neurons > 0:
          synth_shape = (data.shape[0], num_synth_neurons, 1)
          synth_firing_rates = torch.ones(synth_shape) * data.mean()
          synth_spiking = torch.poisson(synth_firing_rates)
          data = torch.cat((data, synth_spiking), dim=1)

        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class AllenTarget:
    def __init__(self, data, id, normalize, normalization_reference=None):
        if normalize:
            if normalization_reference is None:
                data = DataHelpers.normalize_2d(data)
            else:
                mean = normalization_reference.nanmean(axis=0, keepdims=True)
                std = np.nanstd(normalization_reference, axis=0, keepdims=True) + 1e-8
                data = (data - mean) / std
        self.data = data
        self.id = id

    def get_id(self):
        return self.id

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

    def is_eq(self, val):
      return self.data == val


    def not_nan(self):
      return ~torch.isnan(self.data)

class AllenDataset(Dataset):
    def __init__(
        self,
        data_filepath,
        flag,
        dataset_name,
        bin_width,
        input_window,
        output_window,
        normalize_targets,
        num_synth_neurons,
        session_id,
        held_out_visual_stimuli,
        hold_out_visual_stimuli,
    ):

      if (input_window > 1) or (output_window > 1):
        raise UserWarning('Input/output window size > 1 not yet implemented for Allen VC')

      target_names = ['speed', 'condition', 'orientation', 'temporal_frequency']

      self.targets = {}

      self.data_loader = AllenDatasetLoader(
          dataset_name,
          flag,
          data_filepath,
          bin_width,
          session_id
      )
      self.spikes = AllenSpikes(self.data_loader.load_spikes(), num_synth_neurons)

      if self.data_loader.has_pupil is True:
        target_names.append('pupil_size')

      normalization_loader = None
      if normalize_targets and flag != 'train':
        # Normalize test targets with the mean/std of the full training set,
        # before blank filtering or CV subsetting.
        normalization_loader = AllenDatasetLoader(
            dataset_name, 'train', data_filepath, bin_width, session_id
        )

      for target in target_names:
          reference = (
              normalization_loader.load_behaviour(target)
              if normalization_loader is not None else None
          )
          self.targets[target] = AllenTarget(
              self.data_loader.load_behaviour(target), target, normalize_targets,
              normalization_reference=reference,
          )
      self.presentation_ids = self.data_loader.load_metadata('presentation_id')
      self.presentation_bins = self.data_loader.load_metadata('presentation_bin')
      # Keep only samples where orientation is available (not NaN).
      print("Note: Filtering for only samples where drifting grating was shown.")
      valid_targets = self.targets['orientation'].not_nan().squeeze(-1)


      if hold_out_visual_stimuli is True and held_out_visual_stimuli:
        for t_ori, t_freq in zip(held_out_visual_stimuli['orientation'], held_out_visual_stimuli['temporal_frequency']):
          mask = (self.targets['orientation'].is_eq(t_ori) & self.targets['temporal_frequency'].is_eq(t_freq)).squeeze(-1)
          valid_targets = valid_targets & (~mask)

      for target_name in self.targets.keys():
        self.targets[target_name] = self.targets[target_name][valid_targets]

      self.spikes = self.spikes[valid_targets]
      self.presentation_ids = self.presentation_ids[valid_targets]
      self.presentation_bins = self.presentation_bins[valid_targets]

    def __len__(self):
        return len(self.spikes)

    def __getitem__(self, idx):
        item = {}
        item["spikes"] = self.spikes[idx]

        for target in self.targets.keys():
          item[target] = self.targets[target][idx]

        item['sample_index'] = idx
        item['presentation_id'] = self.presentation_ids[idx]
        item['presentation_bin'] = self.presentation_bins[idx]
        return item

    def get_unit_metrics(self):
      return self.data_loader.load_unit_metrics()

    def get_group_ids(self):
      return self.presentation_ids.detach().cpu().numpy()
