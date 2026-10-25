import numpy as np
import torch
from torch import nn
from tqdm import tqdm

class TorchMLPDecoder:
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 15,
        val_fraction: float = 0.1,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model: nn.Module | None = None
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

    def _activation_module(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "gelu":
            return nn.GELU()
        if self.activation == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(self._activation_module())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim == 1:
            y_np = y_np[:, None]

        x_t = torch.from_numpy(x_np).to(self.device)
        y_t = torch.from_numpy(y_np).to(self.device)

        self.x_mean = x_t.mean(dim=0, keepdim=True)
        self.x_std = x_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.y_mean = y_t.mean(dim=0, keepdim=True)
        self.y_std = y_t.std(dim=0, keepdim=True).clamp_min(1e-6)

        x_norm = (x_t - self.x_mean) / self.x_std
        y_norm = (y_t - self.y_mean) / self.y_std

        n = x_norm.shape[0]
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)
        val_n = max(1, int(round(n * self.val_fraction)))
        train_idx = perm[:-val_n] if val_n < n else perm
        val_idx = perm[-val_n:] if val_n < n else perm
        train_idx = train_idx.to(self.device)
        val_idx = val_idx.to(self.device)

        self.model = self._build_model(x_norm.shape[1], y_norm.shape[1])
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        best_epoch = -1
        for epoch in tqdm(range(self.max_epochs), desc="Training Torch Decoder"):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = self.model(x_norm[train_idx])
            train_loss = loss_fn(pred, y_norm[train_idx])
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(x_norm[val_idx])
                val_loss = float(loss_fn(val_pred, y_norm[val_idx]).detach().cpu())
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            elif epoch - best_epoch >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device).eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchMLPDecoder must be fit before predict")
        x_np = np.asarray(x, dtype=np.float32)
        x_t = torch.from_numpy(x_np).to(self.device)
        x_norm = (x_t - self.x_mean) / self.x_std
        with torch.no_grad():
            pred_norm = self.model(x_norm)
            pred = pred_norm * self.y_std + self.y_mean
        return pred.detach().cpu().numpy()
