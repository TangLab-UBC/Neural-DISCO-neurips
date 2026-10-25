import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import Subset

from ..data.rat_in_box_dataset_custom import RatInBoxDatasetCustom

from src.data.allen_dataset import AllenDataset


NLB_DATASETS = {
    "mc_maze",
    "mc_maze_large",
    "mc_maze_medium",
    "mc_maze_small",
    "mc_rtt",
    "area2_bump",
    "dmfc_rsg",
}

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        if_cuda: bool,
        train_batch: int,
        val_batch: int,
        test_batch: int,
        num_workers: int,
        num_folds: int,
        fold_idx: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_samples = None
        self.kwargs = {"pin_memory": True} if self.hparams.if_cuda else {}

        full_dataset = self.dataset("train")

        n = len(full_dataset)
        indices = np.arange(n)
        if hasattr(full_dataset, "get_group_ids"):
            group_ids = np.asarray(full_dataset.get_group_ids())
            unique_groups = np.unique(group_ids)
            if self.hparams.num_folds > len(unique_groups):
                raise ValueError(
                    f"Requested {self.hparams.num_folds} folds but only "
                    f"{len(unique_groups)} dataset groups are available."
                )
            group_folds = np.array_split(unique_groups, self.hparams.num_folds)
            val_groups = group_folds[self.hparams.fold_idx]
            val_mask = np.isin(group_ids, val_groups)
            val_indices = indices[val_mask]
            train_indices = indices[~val_mask]
        else:
            fold_size = n // self.hparams.num_folds
            val_start = self.hparams.fold_idx * fold_size
            val_end = val_start + fold_size if self.hparams.fold_idx < self.hparams.num_folds - 1 else n
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.num_samples = len(self.train_dataset)

        self.test_dataset = self.dataset("test")

        # Each child only has to make self.dataset('test') and self.dataset('val') available.

    def get_num_samples(self):
        if self.num_samples is None:
            train_dataset = self.dataset("train")
            self.num_samples = (
                train_dataset.num_samples if hasattr(train_dataset, "num_samples") else len(train_dataset)
            )
        return self.num_samples

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            **self.kwargs,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.val_batch,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            **self.kwargs,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.test_batch,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            **self.kwargs,
        )


class MCMazeDataModule(DataModule):
    def __init__(
        self,
        input_window: int,
        output_window: int,
        dataset: str,
        if_cuda: bool,
        train_batch: int,
        val_batch: int,
        test_batch: int,
        num_workers: int,
        data_filepath: str,
        smoothing_constant: float,
        normalize_spikes: bool,
        num_folds: int,
        fold_idx: int,
        normalize_targets: bool,
        bin_width: int,
        behaviours: list | None = None,
        target_normalization_mode: str | None = None,
        delay_factor: int = 80,
        unknown_input_window: int | None = None,
        spike_target_mode: str | None = None,
        behaviour_target_mode: str | None = None,
        num_synth_neurons: int = 0,
    ) -> None:

        super().__init__(if_cuda, train_batch, val_batch, test_batch, num_workers, num_folds, fold_idx)
        self.save_hyperparameters()

    def dataset(self, flag):
        from ..data.dataset import NLBDataset
        return NLBDataset(
            data_filepath=self.hparams.data_filepath,
            flag=flag,
            dataset_name=self.hparams.dataset,
            input_window=self.hparams.input_window,
            output_window=self.hparams.output_window,
            smoothing_constant=self.hparams.smoothing_constant,
            normalize_spikes=self.hparams.normalize_spikes,
            normalize_targets=self.hparams.normalize_targets,
            target_normalization_mode=self.hparams.target_normalization_mode,
            delay_factor=self.hparams.delay_factor,
            bin_width=self.hparams.bin_width,
            behaviours=self.hparams.behaviours,
            unknown_input_window=self.hparams.unknown_input_window,
            spike_target_mode=self.hparams.spike_target_mode,
            behaviour_target_mode=self.hparams.behaviour_target_mode,
            num_synth_neurons=self.hparams.num_synth_neurons,
        )


class CustomRatInABoxDataModule(DataModule):
    def __init__(
        self,
        input_window: int,
        output_window: int,
        dataset: str,
        if_cuda: bool,
        train_batch: int,
        val_batch: int,
        test_batch: int,
        num_workers: int,
        data_filepath: str,
        smoothing_constant: float,
        normalize_spikes: bool,
        num_folds: int,
        fold_idx: int,
        normalize_targets: bool,
        bin_width: int,
        num_cells: int,
        noise: float,
        minutes: int,
        num_bins: int,
        use_counterfactual: bool,
        manual_trajectory: bool,
        include_noise_cell: bool,
        delay_factor: int = 80,
        unknown_input_window: int | None = None,
        spike_target_mode: str | None = None,
        behaviour_target_mode: str | None = None,
    ) -> None:
        super().__init__(if_cuda, train_batch, val_batch, test_batch, num_workers, num_folds, fold_idx)
        self.save_hyperparameters()

    def dataset(self, flag):
        return RatInBoxDatasetCustom(
            data_filepath=self.hparams.data_filepath,
            flag=flag,
            dataset_name=self.hparams.dataset,
            input_window=self.hparams.input_window,
            output_window=self.hparams.output_window,
            smoothing_constant=self.hparams.smoothing_constant,
            normalize_spikes=self.hparams.normalize_spikes,
            normalize_targets=self.hparams.normalize_targets,
            delay_factor=self.hparams.delay_factor,
            bin_width=self.hparams.bin_width,
            num_cells=self.hparams.num_cells,
            noise=self.hparams.noise,
            minutes=self.hparams.minutes,
            num_bins=self.hparams.num_bins,
            use_counterfactual=self.hparams.use_counterfactual,
            manual_trajectory=self.hparams.manual_trajectory,
            include_noise_cell=self.hparams.include_noise_cell,
            unknown_input_window=self.hparams.unknown_input_window,
            spike_target_mode=self.hparams.spike_target_mode,
            behaviour_target_mode=self.hparams.behaviour_target_mode,
        )

class AllenDataModule(DataModule):
    def __init__(
        self,
        dataset: str,
        if_cuda: bool,
        train_batch: int,
        val_batch: int,
        test_batch: int,
        num_workers: int,
        data_filepath: str,
        num_folds: int,
        fold_idx: int,
        input_window: int,
        output_window: int,
        normalize_targets: bool,
        bin_width: int,
        num_synth_neurons: int,
        session_id: int,
        held_out_visual_stimuli: dict,
    ) -> None:
        super().__init__(if_cuda, train_batch, val_batch, test_batch, num_workers, num_folds, fold_idx)
        self.save_hyperparameters()

    def dataset(self, flag, hold_out_visual_stimuli=True):
        dataset = AllenDataset(
            data_filepath=self.hparams.data_filepath,
            flag=flag,
            input_window=self.hparams.input_window,
            output_window=self.hparams.output_window,
            dataset_name=self.hparams.dataset,
            bin_width=self.hparams.bin_width,
            normalize_targets=self.hparams.normalize_targets,
            num_synth_neurons=self.hparams.num_synth_neurons,
            session_id=self.hparams.session_id,
            held_out_visual_stimuli=self.hparams.held_out_visual_stimuli,
            hold_out_visual_stimuli=hold_out_visual_stimuli
        )
        return dataset
