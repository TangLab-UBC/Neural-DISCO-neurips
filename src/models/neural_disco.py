import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch import nn
from tqdm import tqdm

from .components import MLP, MLPDecoder, MLPEncoder


class BehaviourEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        behaviour_dim: int,
        num_layers: int,
    ):
        super().__init__()
        encoder_dims = list(np.linspace(behaviour_dim, latent_dim, num_layers, dtype=int))
        self.encoder = MLP(encoder_dims)

    def forward(self, x):
        return self.encoder(x)

    def l1(self):
        return self.encoder.l1()


class NeuralDISCO(nn.Module):
    def __init__(
        self,
        behaviour_latent_dim: int,
        input_window: int,
        output_window: int,
        num_layers: int,
        num_neurons: int,
        behaviours: list,
        behaviour_dims: dict,
        behaviour_model_types: dict | None,
        behaviour_num_layers: int | None,
        unknown_num_layers: int | None,
        decoder_num_layers: int | None,
        unknown_attribute_latent_dim: int,
        unknown_sigma: float,
        mlp: bool = True,
        unknown_model_type: str | None = None,
        unknown_input_window: int | None = None,
        always_zero_unknown: bool = False,
        unknown_input_key: str = "spikes",
    ) -> None:
        super().__init__()
        if not mlp:
            raise ValueError("This codebase supports the MLP Neural-DISCO architecture only.")
        if unknown_model_type not in (None, "mlp"):
            raise ValueError(f"Unsupported unknown_model_type: {unknown_model_type}")

        self.mlp = True
        self.behaviour_models = nn.ModuleList()
        self.behaviours = behaviours
        self.behaviour_dims = behaviour_dims
        self.behaviour_model_types = behaviour_model_types or {}
        self.behaviour_latent_dim = behaviour_latent_dim
        self.num_behaviours = len(behaviours)
        self.unknown_sigma = unknown_sigma
        self.num_neurons = num_neurons
        self.output_window = output_window
        self.unknown_attribute_latent_dim = unknown_attribute_latent_dim
        self.train_encoders = False
        self.embed = False
        self.force_zero_unknown = False
        self.always_zero_unknown = always_zero_unknown
        self.unknown_input_key = unknown_input_key
        self.unknown_input_window = unknown_input_window or input_window
        self.behaviour_num_layers = behaviour_num_layers or num_layers
        self.unknown_num_layers = unknown_num_layers or num_layers
        self.decoder_num_layers = decoder_num_layers or num_layers

        for behaviour in behaviours:
            model_type = self.behaviour_model_types.get(behaviour, "mlp")
            if model_type != "mlp":
                raise ValueError(
                    f"Unsupported behaviour model type for {behaviour}: {model_type}"
                )
            self.behaviour_models.append(
                BehaviourEncoder(
                    behaviour_latent_dim,
                    behaviour_dim=self.behaviour_dims[behaviour],
                    num_layers=self.behaviour_num_layers,
                )
            )

        total_latent_dim = (
            self.num_behaviours * behaviour_latent_dim
            + unknown_attribute_latent_dim
        )
        self.unknown_attribute_encoder = MLPEncoder(
            latent_dim=unknown_attribute_latent_dim,
            num_layers=self.unknown_num_layers,
            input_frame_dim=num_neurons,
            input_window=self.unknown_input_window,
            dimension_shift="linspace",
        )
        self.reconstruction_decoder = MLPDecoder(
            latent_dim=total_latent_dim,
            num_layers=self.decoder_num_layers,
            output_frame_dim=num_neurons,
            output_window=output_window,
        )
        self.soft_plus = nn.Softplus()

    def forward(self, batch):
        output_dict = {}
        latents = [model(batch[behaviour]) for behaviour, model in zip(self.behaviours, self.behaviour_models)]

        unknown_attribute_latent = self.get_unknown_latent(batch)
        if self.force_zero_unknown or self.always_zero_unknown:
            unknown_attribute_latent = torch.zeros_like(unknown_attribute_latent)

        if self.train_encoders:
            embedding_target = self.unknown_attribute_embedding[batch["sample_index"]]
            output_dict["embedding_target"] = embedding_target
            output_dict["encoder_embedding"] = unknown_attribute_latent

        latents.append(unknown_attribute_latent)
        output_dict["l2_unknown"] = unknown_attribute_latent.pow(2).sum(dim=1).mean()
        combined_latent = torch.cat(latents, dim=1)
        reconstruction_prediction = self.soft_plus(self.reconstruction_decoder(combined_latent))
        output_dict["reconstruction"] = reconstruction_prediction.view(
            -1, self.num_neurons, self.output_window
        )
        return output_dict

    def get_unknown_latent(self, batch):
        if self.embed:
            embedding = self.unknown_attribute_embedding(batch["sample_index"])
        else:
            embedding = self.unknown_attribute_encoder(batch[self.unknown_input_key])

        if self.training and self.unknown_sigma > 0:
            embedding = embedding + torch.randn_like(embedding) * self.unknown_sigma
        return embedding

    def attribute(self, batch):
        return self.attribute_ig(batch)

    def attribute_ig(self, batch):
        behaviour_dims = []
        behaviour_indices = {}
        output_dict = {"unknown": torch.zeros(self.num_neurons)}

        for behaviour in self.behaviours:
            dims = self.behaviour_dims[behaviour]
            dims_prev = sum(behaviour_dims)
            behaviour_indices[behaviour] = list(range(dims_prev, dims_prev + dims))
            behaviour_dims.append(dims)

        total_behaviour_dim = sum(behaviour_dims)
        batch_size = len(batch["spikes"])
        behaviour_input = torch.zeros((batch_size, total_behaviour_dim))

        for behaviour in self.behaviours:
            output_dict[behaviour] = torch.zeros(self.num_neurons)
            indices = behaviour_indices[behaviour]
            behaviour_input[:, indices] = batch[behaviour].detach().requires_grad_(True)

        def ig_model_wrapper(behaviour_input, unknown_attribute_latent):
            latents = []
            for behaviour, model in zip(self.behaviours, self.behaviour_models):
                indices = behaviour_indices[behaviour]
                latents.append(model(behaviour_input[:, indices]))
            latents.append(unknown_attribute_latent)
            combined_latent = torch.cat(latents, dim=1)
            return self.reconstruction_decoder(combined_latent)

        ig = IntegratedGradients(ig_model_wrapper)
        unknown_attribute_latent = self.get_unknown_latent(batch).detach().requires_grad_(True)

        for neuron_idx in tqdm(range(self.num_neurons), desc="Attributing neurons"):
            attributions = ig.attribute(
                inputs=(behaviour_input, unknown_attribute_latent),
                target=neuron_idx,
            )

            behaviour_attrs = attributions[0].mean(dim=0)
            unknown_attr = attributions[1].mean(dim=0)
            output_dict["unknown"][neuron_idx] = unknown_attr.abs().mean().detach()
            for behaviour in self.behaviours:
                indices = behaviour_indices[behaviour]
                behavioural_attr = behaviour_attrs[indices]
                output_dict[behaviour][neuron_idx] = behavioural_attr.abs().mean().detach()

        return output_dict

    def attribute_grad(self, batch):
        behaviour_tensors = []
        output_dict = {"unknown": torch.zeros(self.num_neurons)}

        for behaviour in self.behaviours:
            tensor = batch[behaviour].detach().requires_grad_(True)
            behaviour_tensors.append(tensor)
            output_dict[behaviour] = torch.zeros(self.num_neurons)

        latents = [model(tensor) for tensor, model in zip(behaviour_tensors, self.behaviour_models)]
        unknown_attribute_latent = self.get_unknown_latent(batch).detach().requires_grad_(True)
        latents.append(unknown_attribute_latent)
        combined_latent = torch.cat(latents, dim=1)
        output = self.reconstruction_decoder(combined_latent).sum(dim=0)

        for neuron_idx in range(self.num_neurons):
            grads = torch.autograd.grad(
                output[neuron_idx],
                behaviour_tensors,
                retain_graph=True,
            )
            neuron_raw_grads = {
                behaviour: grad.mean(dim=1).abs().mean()
                for behaviour, grad in zip(self.behaviours, grads)
            }
            unknown_grad = torch.autograd.grad(
                output[neuron_idx],
                unknown_attribute_latent,
                retain_graph=True,
            )[0]
            neuron_raw_grads["unknown"] = unknown_grad.mean(dim=1).abs().mean()
            total_grads = sum(neuron_raw_grads.values())
            for key, value in neuron_raw_grads.items():
                output_dict[key][neuron_idx] = value / total_grads
        return output_dict

    def l1(self, data=None):
        penalties = [model.l1() for model in self.behaviour_models]
        penalties.append(self.unknown_attribute_encoder.l1())
        return sum(penalties)

    def set_train_encoders(self):
        embeddings = self.unknown_attribute_embedding.weight.detach().clone()
        del self.unknown_attribute_embedding
        self.unknown_attribute_embedding = embeddings
        self.embed = False
        self.train_encoders = True

    def set_train_embeddings(self, num_samples):
        self.embed = True
        self.unknown_attribute_embedding = nn.Embedding(
            num_embeddings=num_samples,
            embedding_dim=self.unknown_attribute_latent_dim,
        )
