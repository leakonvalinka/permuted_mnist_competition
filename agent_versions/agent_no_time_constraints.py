# doesnt work

import numpy as np
import torch
import torch.nn as nn


_TORCH_THREADS_CONFIGURED = False

def _configure_torch_threads_once(num_threads: int = 2, num_interop_threads: int = 1) -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        # Can happen if parallel work has already started in this process.
        pass

    _TORCH_THREADS_CONFIGURED = True


class MLP(nn.Module):
    def __init__(self, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    """
    CPU-only agent with no train/predict deadline.
    Only `safety_margin_s` is kept as a configurable time-related setting.
    """

    def __init__(
        self,
        seed: int = None,
        batch_size: int = 2048,
        max_epochs: int = 10_000,
        lr: float = 1.5e-3,
        weight_decay: float = 5e-5,
        dropout: float = 0.05,
        safety_margin_s: float = 1.5,
        predict_batch_size: int = 4096,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        # Enforce "2 cores" inside the process (important under Docker too)
        _configure_torch_threads_once(num_threads=2, num_interop_threads=1)

        self.device = torch.device("cpu")
        self.model = MLP(dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.safety_margin_s = safety_margin_s
        self.predict_batch_size = predict_batch_size

    def _prep_x(self, X: np.ndarray) -> torch.Tensor:
        # Convert once; keep contiguous float32
        x = torch.from_numpy(X.reshape(len(X), -1)).to(dtype=torch.float32)
        x = x / 255.0
        # MNIST standardization (fast, helps convergence)
        x = (x - 0.1307) / 0.3081
        return x

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X = self._prep_x(X_train).to(self.device)
        y = torch.from_numpy(y_train.ravel().astype(np.int64)).to(self.device)

        n = X.shape[0]
        self.model.train()

        for _epoch in range(self.max_epochs):
            perm = torch.randperm(n)

            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = X[idx]
                yb = y[idx]

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = self._prep_x(X_test).to(self.device)
        self.model.eval()

        # Chunked prediction => more stable memory/time
        bs = self.predict_batch_size
        preds = []

        for i in range(0, X.shape[0], bs):
            logits = self.model(X[i : i + bs])
            preds.append(logits.argmax(dim=1).cpu())

        return torch.cat(preds).numpy()