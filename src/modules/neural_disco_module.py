from src.constants import DEFAULT_BEHAVIOUR_DIMS
import torch
from torch import nn
import pytorch_lightning as pl

from ..models.neural_disco import NeuralDISCO
from sklearn.metrics import r2_score


class NeuralDISCOModule(pl.LightningModule):
    def __init__(
        self,
        input_window: int,
        output_window: int,
        num_layers: int,
        num_neurons: int,
        latent_dim: int,
        unknown_attribute_latent_dim: int,
        unknown_sigma: float,
        l1_alpha: float,
        l2_unknown: float,
        lr: float,
        if_cuda: bool,
        gamma: float,
        weight_decay: float,
        reconstruction_style: bool,
        mlp: bool,
        dataset: str,
        reconstruction_target_key: str = "spikes",
        encoder_loss_alpha: float = 1.0,
        reconstruction_penalty: float = 0.0,
        l2_warmup_epochs: int = 0,
        stage0_unknown_off_epochs: int = 0,
        always_zero_unknown: bool = False,
        unknown_input_key: str = "spikes",
        unknown_model_type: str | None = None,
        unknown_input_window: int | None = None,
        num_workers: int = 8,
        lr_schedule: list | None = None,
        behaviours: list | None = None,
        behaviour_dims: dict | None = None,
        behaviour_model_types: dict | None = None,
        behaviour_num_layers: int | None = None,
        unknown_num_layers: int | None = None,
        decoder_num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = (
            {"num_workers": self.hparams.num_workers, "pin_memory": True}
            if self.hparams.if_cuda
            else {}
        )

        if dataset in DEFAULT_BEHAVIOUR_DIMS:
            default_behaviours = DEFAULT_BEHAVIOUR_DIMS[dataset].keys()
            default_behaviour_dims = DEFAULT_BEHAVIOUR_DIMS[dataset]
        else:
            if behaviours is None:
                raise ValueError("No behaviour defaults available. Must supply behaviours in config.")

        if behaviour_dims is not None:
            self.behaviours = list(behaviours) if behaviours is not None else list(behaviour_dims.keys())
            missing_dims = [
                behaviour for behaviour in self.behaviours if behaviour not in behaviour_dims
            ]
            if missing_dims:
                raise ValueError(f"Missing behaviour dims for: {missing_dims}")
            self.behaviour_dims = {
                behaviour: behaviour_dims[behaviour] for behaviour in self.behaviours
            }
        elif behaviours is None:
            self.behaviours = list(default_behaviours)
            self.behaviour_dims = {
                behaviour: default_behaviour_dims[behaviour] for behaviour in self.behaviours
            }
        else:
            unknown_behaviours = set(behaviours) - set(default_behaviours)
            if unknown_behaviours:
                raise ValueError(f"Unsupported behaviours for dataset: {sorted(unknown_behaviours)}")
            self.behaviours = list(behaviours)
            self.behaviour_dims = {
                behaviour: default_behaviour_dims[behaviour] for behaviour in self.behaviours
            }

        self.loss = nn.MSELoss()
        self.train_encoders = False

        if self.hparams.reconstruction_style == "Poisson":
            self.reconstruction_loss = nn.PoissonNLLLoss(log_input=False)
        elif self.hparams.reconstruction_style == "MSE":
            self.reconstruction_loss = nn.MSELoss()
        else:
            raise ValueError('Reconstruction style must be one of "Poisson" and "MSE"')

        self.encoder_loss = nn.MSELoss()
        self.model = NeuralDISCO(
            behaviour_latent_dim=latent_dim,
            input_window=input_window,
            output_window=output_window,
            num_layers=num_layers,
            num_neurons=num_neurons,
            behaviours=self.behaviours,
            behaviour_dims=self.behaviour_dims,
            behaviour_model_types=behaviour_model_types,
            behaviour_num_layers=behaviour_num_layers,
            unknown_num_layers=unknown_num_layers,
            decoder_num_layers=decoder_num_layers,
            unknown_attribute_latent_dim=unknown_attribute_latent_dim,
            unknown_sigma=unknown_sigma,
            mlp=mlp,
            unknown_model_type=unknown_model_type,
            unknown_input_window=unknown_input_window,
            always_zero_unknown=always_zero_unknown,
            unknown_input_key=unknown_input_key,
        )

    @staticmethod
    def bits_per_spike(pred_rates, spikes):
        eps = 1e-8
        pred_rates = pred_rates.detach().clamp_min(eps)
        spikes = spikes.detach()
        null_rates = spikes.mean(dim=0, keepdim=True).clamp_min(eps)
        loglik_model = (spikes * torch.log(pred_rates) - pred_rates).sum()
        loglik_null = (spikes * torch.log(null_rates) - null_rates).sum()
        denom = spikes.sum().clamp_min(eps)
        return (loglik_model - loglik_null) / (
            denom * torch.log(torch.tensor(2.0, device=pred_rates.device))
        )

    def training_step(self, batch, batch_idx):
        self.model.force_zero_unknown = self.current_epoch < self.hparams.stage0_unknown_off_epochs
        losses = self.step(batch, compute_bps=False)
        train_loss = losses["reconstruction"]
        self.log(
            "train_reconstruction_loss",
            losses["reconstruction"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if "reconstruction_mse" in losses and self.hparams.reconstruction_penalty > 0:
            train_loss += self.hparams.reconstruction_penalty * losses["reconstruction_mse"]
            self.log(
                "train_reconstruction_mse",
                losses["reconstruction_mse"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        l2_weight = self.hparams.l2_unknown
        if self.hparams.l2_warmup_epochs > 0:
            warmup_frac = min(
                1.0,
                float(self.current_epoch + 1) / float(self.hparams.l2_warmup_epochs),
            )
            l2_weight = l2_weight * warmup_frac
        train_loss += losses["l2_unknown"] * l2_weight
        self.log("train_l2_weight", l2_weight, on_step=False, on_epoch=True, logger=True)

        if self.hparams.l1_alpha > 0:
            train_loss += self.hparams.l1_alpha * losses["l1"]

        if "encoder_train_loss" in losses:
            train_loss += self.hparams.encoder_loss_alpha * losses["encoder_train_loss"]
            self.log(
                "encoder_train_loss",
                losses["encoder_train_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        self.log("train_loss", train_loss, on_step=False, on_epoch=True, logger=True)
        return train_loss

    def step(self, batch, compute_bps=False):
        losses = {}
        target_key = self.hparams.reconstruction_target_key
        if target_key not in batch:
            raise KeyError(f"Missing reconstruction target '{target_key}' in batch")
        data = batch[target_key]

        output_dict = self.model(batch)
        if self.hparams.l1_alpha > 0:
            losses["l1"] = self.model.l1(data)

        losses["reconstruction"] = self.reconstruction_loss(output_dict["reconstruction"], data)
        losses["reconstruction_mse"] = self.loss(output_dict["reconstruction"], data)
        losses["l2_unknown"] = output_dict["l2_unknown"]

        if self.train_encoders:
            losses["encoder_train_loss"] = self.encoder_loss(
                output_dict["encoder_embedding"], output_dict["embedding_target"]
            )

        if compute_bps and target_key == "spikes":
            losses["bps"] = self.bits_per_spike(output_dict["reconstruction"], data)

        if "rates" in batch:
            pred_rates = output_dict["reconstruction"].detach().squeeze().float().cpu().numpy()
            rates = batch["rates"].detach().squeeze().float().cpu().numpy()
            losses["rates_r2"] = r2_score(rates, pred_rates)

        return losses

    def validation_step(self, batch, batch_idx):
        self.model.force_zero_unknown = False
        losses = self.step(batch, compute_bps=True)
        val_loss = losses["reconstruction"]
        self.log(
            "val_reconstruction_loss",
            losses["reconstruction"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if "rates_r2" in losses:
            self.log(
                "rates_r2",
                losses["rates_r2"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        if "bps" in losses:
            self.log(
                "val_bps",
                losses["bps"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        milestones = self.hparams.lr_schedule or []
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def set_train_encoders(self):
        self.train_encoders = True
        self.model.set_train_encoders()

    def set_train_embeddings(self, num_samples):
        self.model.set_train_embeddings(num_samples)
