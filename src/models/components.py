import numpy as np
import torch
from torch import nn


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, output_frame_dim, output_window, num_layers):
        super().__init__()
        out_dim = output_frame_dim * output_window
        self.output_frame_dim = output_frame_dim
        self.output_window = output_window
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()

        decoder_dims = list(np.linspace(latent_dim, out_dim, num_layers, dtype=int))
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))

    def decoder(self, latent):
        x = latent
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def forward(self, x):
        out = self.decoder(x)
        return out.reshape(-1, self.output_frame_dim, self.output_window)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, latent):
        x = latent
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def l1(self):
        if not self.layers:
            return torch.tensor(0.0)
        return sum(param.abs().sum() for param in self.parameters())


class MLPEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_frame_dim,
        input_window,
        dimension_shift,
        num_layers,
    ):
        super().__init__()
        if dimension_shift not in ["linspace", "constant"]:
            raise ValueError('dimension_shift must be one of "constant" or "linspace"')

        self.input_window = input_window
        if input_window > 1:
            self.linear_temporal_projection_layer = TemporalProjectionLayer(
                input_frame_dim, input_window
            )

        if dimension_shift == "linspace":
            encoder_dims = list(
                np.linspace(input_frame_dim, latent_dim, num_layers, dtype=int)
            )
        else:
            encoder_dims = np.full(num_layers, latent_dim, dtype=int)
            encoder_dims[0] = input_frame_dim
            encoder_dims = list(encoder_dims)

        self.encoder = MLP(encoder_dims)

    def forward(self, x):
        if self.input_window > 1:
            per_neuron_projection = self.linear_temporal_projection_layer(x)
        else:
            per_neuron_projection = x.squeeze(-1)
        return self.encoder(per_neuron_projection)

    def l1(self):
        return self.encoder.l1()


class TemporalProjectionLayer(nn.Module):
    def __init__(self, num_neurons, num_timepoints):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_neurons, num_timepoints))
        self.b = nn.Parameter(torch.zeros(num_neurons))

    def forward(self, x):
        out = torch.sum(x * self.W, dim=2)
        return out + self.b
