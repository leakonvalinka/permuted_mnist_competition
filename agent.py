import time
import numpy as np
import torch
import torch.nn as nn


_threads_configured = False

def _set_resource_limits(num_threads: int = 2, num_interop_threads: int = 1) -> None:
    global _threads_configured
    if _threads_configured:
        return

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        # Can happen if parallel work has already started in this process.
        pass

    _threads_configured = True


class MLP(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(
        self,
        seed: int = None,
        batch_size: int = 512,
        max_epochs: int = 10_000,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        aug_noise: float = 0.5,       # Gaussian noise std added to inputs during training
        episode_budget_s: float = 60.0,
        safety_margin_s: float = 2.5, # time buffer (includes predict + data prep)
        predict_batch_size: int = 4096,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        # enforce "2 cores"
        _set_resource_limits(num_threads=2, num_interop_threads=1)

        self.device = torch.device("cpu")
        self.model = MLP(dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Cosine annealing with warm restarts to escape local minima
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.aug_noise = aug_noise

        self.episode_budget_s = episode_budget_s
        self.safety_margin_s = safety_margin_s
        self.predict_batch_size = predict_batch_size

    def _prep_x(self, X: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(X.reshape(len(X), -1)).to(dtype=torch.float32)
        x = x / 255.0
        # mnist standardization
        x = (x - 0.1307) / 0.3081
        return x

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        t0 = time.perf_counter()

        train_deadline = t0 + (self.episode_budget_s - self.safety_margin_s)
        if train_deadline <= t0:
            return

        X = self._prep_x(X_train).to(self.device)
        y = torch.from_numpy(y_train.ravel().astype(np.int64)).to(self.device)

        n = X.shape[0]
        self.model.train()

        for epoch in range(self.max_epochs):
            perm = torch.randperm(n)

            for i in range(0, n, self.batch_size):
                if time.perf_counter() >= train_deadline:
                    return

                idx = perm[i : i + self.batch_size]
                xb = X[idx]
                # Gaussian noise augmentation: regularises and matches eval-time noise
                if self.aug_noise > 0.0:
                    xb = xb + torch.randn_like(xb) * self.aug_noise

                logits = self.model(xb)
                loss = self.criterion(logits, y[idx])

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step(epoch)

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = self._prep_x(X_test).to(self.device)
        self.model.eval()
        bs = self.predict_batch_size
        preds = []

        for i in range(0, X.shape[0], bs):
            logits = self.model(X[i : i + bs])
            preds.append(logits.argmax(dim=1).cpu())

        return torch.cat(preds).numpy()
