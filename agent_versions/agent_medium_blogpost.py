# doenst work

"""
CNN-based agent following the approach from the Medium article:
"How I Hit 99.26% Accuracy on MNIST with a CNN in PyTorch"
by Amit Subhash Chejara (Data Science Collective, 2025).
https://medium.com/data-science-collective/implementing-cnn-in-pytorch-testing-on-mnist-99-26-test-accuracy-5c63876c6ac8

Architecture:
    ZeroPad2d(2) → Conv2d(1, 16, 5, stride=1) → BatchNorm2d(16) → ReLU
    → MaxPool2d(2) → Flatten → LazyLinear(10) → Softmax

Training:
    Optimizer : Adam (lr=0.01)
    Scheduler : LinearLR  (linearly decays LR each epoch)
    Loss      : CrossEntropyLoss
    Batch size: 32

Time-budget adaptation:
    Training is stopped early if the episode wall-clock budget is about to
    expire (same guard as the reference agent.py).
"""

import time
import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Exact Sequential architecture from the article's `my_beloved_model()`.

    Input  : (N, 1, 28, 28)  — grayscale images scaled to [0, 1]
    Output : (N, 10)          — class probabilities via Softmax
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ZeroPad2d(2),              # 28×28 → 32×32
            nn.Conv2d(1, 16, 5, 1),       # → (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),              # → (16, 14, 14)
            nn.Flatten(),                 # → 3 136
            nn.LazyLinear(out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    def __init__(
        self,
        seed: int = None,
        batch_size: int = 32,           # article: batch_size = 32
        max_epochs: int = 10_000,       # effectively capped by the time budget
        lr: float = 0.01,              # article: lr = 0.01 with Adam
        episode_budget_s: float = 60.0,
        safety_margin_s: float = 1.5,  # buffer so we never exceed 60 s
        predict_reserve_s: float = 4.0, # headroom for predict()
        predict_batch_size: int = 4096,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.model = CNN().to(self.device)

        # LazyLinear only materialises its weight matrix on the *first*
        # forward pass.  We must do that before constructing the optimizer,
        # otherwise the new parameters are invisible to it.
        with torch.no_grad():
            self.model(torch.zeros(1, 1, 28, 28, device=self.device))

        self.criterion = nn.CrossEntropyLoss()  # article uses CrossEntropyLoss
        self.optimizer = torch.optim.Adam(      # article uses Adam
            self.model.parameters(), lr=lr
        )
        # article wraps Adam with a LinearLR scheduler (steps each epoch)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.episode_budget_s = episode_budget_s
        self.safety_margin_s = safety_margin_s
        self.predict_reserve_s = predict_reserve_s
        self.predict_batch_size = predict_batch_size

    def _prep_x(self, X: np.ndarray) -> torch.Tensor:
        """
        Convert raw numpy input to a (N, 1, 28, 28) float32 tensor scaled to
        [0, 1] — equivalent to torchvision's ToTensor() used in the article.
        """
        x = torch.from_numpy(X.reshape(len(X), 1, 28, 28)).to(dtype=torch.float32)
        x = x / 255.0
        return x

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        t0 = time.perf_counter()
        train_deadline = t0 + (
            self.episode_budget_s - self.safety_margin_s - self.predict_reserve_s
        )
        if train_deadline <= t0:
            return

        X = self._prep_x(X_train).to(self.device)
        y = torch.from_numpy(y_train.ravel().astype(np.int64)).to(self.device)

        n = X.shape[0]
        self.model.train()

        for _epoch in range(self.max_epochs):
            perm = torch.randperm(n)

            for i in range(0, n, self.batch_size):
                if time.perf_counter() >= train_deadline:
                    return

                idx = perm[i : i + self.batch_size]
                xb = X[idx]
                yb = y[idx]

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            # Step the LR scheduler once per completed epoch (as in the article)
            self.scheduler.step()

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
