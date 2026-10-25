from src.constants import DEFAULT_BEHAVIOUR_DIMS
import pytorch_lightning as pl
import torch
from torch import nn

from ..models.gaussian_glm import GaussianGLM


class GaussianGLMModule(pl.LightningModule):
    """Gaussian GLM Lightning module."""

    def __init__(
        self,
        input_window: int,
        output_window: int,
        lr: float,
        if_cuda: bool,
        gamma: float,
        weight_decay: float,
        dataset: str,
        num_neurons: int,
        reconstruction_target_key: str = "spikes",
        num_workers: int = 8,
        lr_schedule: list | None = None,
        behaviours: list | None = None,
        behaviour_dims: dict | None = None,
    ) -> None:
        super().__init__()
        if input_window != 1:
            raise ValueError("GaussianGLMModule only supports input_window=1")
        if output_window != 1:
            raise ValueError("GaussianGLMModule only supports output_window=1")

        self.save_hyperparameters()
        self.kwargs = (
            {"num_workers": self.hparams.num_workers, "pin_memory": True}
            if self.hparams.if_cuda
            else {}
        )

        if dataset in DEFAULT_BEHAVIOUR_DIMS:
            default_behaviours = list(DEFAULT_BEHAVIOUR_DIMS[dataset].keys())
            default_behaviour_dims = DEFAULT_BEHAVIOUR_DIMS[dataset]
        else:
            if behaviours is None:
                raise ValueError(
                    "No behaviour defaults available. Must supply behaviours in config."
                )

        if behaviour_dims is not None:
            self.behaviours = (
                list(behaviours)
                if behaviours is not None
                else list(behaviour_dims.keys())
            )
            missing_dims = [
                behaviour
                for behaviour in self.behaviours
                if behaviour not in behaviour_dims
            ]
            if missing_dims:
                raise ValueError(f"Missing behaviour dims for: {missing_dims}")
            self.behaviour_dims = {
                behaviour: behaviour_dims[behaviour]
                for behaviour in self.behaviours
            }
        elif behaviours is None:
            self.behaviours = default_behaviours
            self.behaviour_dims = {
                behaviour: default_behaviour_dims[behaviour]
                for behaviour in self.behaviours
            }
        else:
            unknown_behaviours = set(behaviours) - set(default_behaviours)
            if unknown_behaviours:
                raise ValueError(
                    f"Unsupported behaviours for dataset: {sorted(unknown_behaviours)}"
                )
            self.behaviours = list(behaviours)
            self.behaviour_dims = {
                behaviour: default_behaviour_dims[behaviour]
                for behaviour in self.behaviours
            }

        self.loss = nn.MSELoss()
        self.model = GaussianGLM(
            num_neurons=num_neurons,
            behaviours=self.behaviours,
            behaviour_dims=self.behaviour_dims,
        )

    def training_step(self, batch, batch_idx):
        losses = self.step(batch)
        self.log(
            "train_loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch)
        self.log(
            "val_loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return losses["loss"]

    def test_step(self, batch, batch_idx):
        losses = self.step(batch)
        self.log(
            "test_loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return losses["loss"]

    def step(self, batch):
        target_key = self.hparams.reconstruction_target_key
        if target_key not in batch:
            raise KeyError(f"Missing reconstruction target '{target_key}' in batch")
        target = batch[target_key]
        prediction = self.model(batch)["reconstruction"]
        loss = self.loss(prediction, target)
        return {"loss": loss, "prediction": prediction, "target": target}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.lr_schedule:
            return optimizer
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.hparams.lr_schedule),
            gamma=self.hparams.gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
