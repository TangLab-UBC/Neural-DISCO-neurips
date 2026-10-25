import torch
from torch import nn


class GaussianGLM(nn.Module):
    """One-layer identity-link Gaussian GLM."""

    def __init__(
        self,
        num_neurons: int,
        behaviours: list,
        behaviour_dims: dict,
    ) -> None:
        super().__init__()
        self.behaviours = behaviours
        self.behaviour_dims = behaviour_dims
        self.input_dim = int(
            sum(self.behaviour_dims[behaviour] for behaviour in self.behaviours)
        )
        self.linear = nn.Linear(self.input_dim, num_neurons, bias=True)

    def forward(self, batch):
        inputs = [
            batch[behaviour].reshape(batch[behaviour].shape[0], -1)
            for behaviour in self.behaviours
        ]
        x = torch.cat(inputs, dim=1)
        reconstruction = self.linear(x)
        return {"reconstruction": reconstruction.unsqueeze(-1)}
