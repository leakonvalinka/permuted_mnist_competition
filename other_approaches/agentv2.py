"""
AgentV2: Neural Feature Extractor + Gradient Boosting Head
============================================================

After substantial empirical exploration (batch size tuning, Lion optimizer,
kNN ensemble, TTA, deep ensemble, pseudo-labels), the supervised MLP
(784→512→512→256→10, batch=512, AdamW+cosine LR) appears to saturate at
~98.74% with a pure backprop approach.

This agent combines neural feature learning with a complementary tree-based
classifier:

Phase 1 (~40s): Train the MLP on all 60k samples.  Instead of using the
  final softmax layer, we use the 256-dimensional penultimate layer as a
  learned feature extractor.  The MLP weights are trained end-to-end with
  standard cross-entropy as usual.

Phase 2 (~15s): Extract 256-dim embeddings for all 60k training samples,
  then fit a **HistGradientBoostingClassifier** (sklearn's fast GBDT, ~O(n·bins)
  complexity) on those features.  GBDT can capture decision boundaries that
  the MLP's softmax layer cannot (non-linear combinations of features,
  interactions that BatchNorm / Dropout regularise away).

Prediction: ensemble MLP softmax + GBDT softmax (probability output) with a
  tuned blend weight (0.7 MLP + 0.3 GBDT by default).

Why GBDT on neural features beats raw GBDT:
  - Raw pixel GBDT on permuted MNIST ignores the consistent permutation.
  - Neural features encode the task-specific class structure learned by the MLP.
  - The 256-d embedding is much denser and informative than 784 raw pixels.

Research basis:
  - "Tabular Data: Deep Learning is Not All You Need" (Shwartz-Ziv & Armon 2022)
    shows GBDT often beats deep nets on structured data; combining them helps.
  - Neural Oblivious Decision Ensembles (Popov et al., 2019) shows tree+neural
    hybrids consistently outperform either alone on tabular tasks.
"""

import time
import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Thread configuration
# ---------------------------------------------------------------------------

_threads_configured = False


def _set_resource_limits(num_threads: int = 2, num_interop_threads: int = 1) -> None:
    global _threads_configured
    if _threads_configured:
        return
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        pass
    _threads_configured = True


# ---------------------------------------------------------------------------
# Model with feature extractor hook
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """784 → 512 → 512 → 256 → 10.  penultimate() returns 256-dim features."""

    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.backbone = nn.Sequential(
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
        )
        self.head = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim penultimate features."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        seed: int = None,
        dropout: float = 0.05,
        batch_size: int = 512,
        max_epochs: int = 10_000,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        episode_budget_s: float = 60.0,
        safety_margin_s: float = 3.0,
        # Phase split: fraction of budget for MLP training
        mlp_frac: float = 0.70,
        # GBDT hyperparameters (fast HistGBDT)
        gbdt_max_iter: int = 100,
        gbdt_max_leaf_nodes: int = 31,
        gbdt_learning_rate: float = 0.1,
        # Ensemble blend weights
        mlp_blend: float = 0.7,
        gbdt_blend: float = 0.3,
        predict_batch_size: int = 4096,
        embed_batch_size: int = 4096,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        _set_resource_limits(num_threads=2, num_interop_threads=1)

        self.device = torch.device("cpu")
        self.model = MLP(dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )

        self.batch_size = batch_size
        self.episode_budget_s = episode_budget_s
        self.safety_margin_s = safety_margin_s
        self.mlp_frac = mlp_frac
        self.gbdt_max_iter = gbdt_max_iter
        self.gbdt_max_leaf_nodes = gbdt_max_leaf_nodes
        self.gbdt_learning_rate = gbdt_learning_rate
        self.mlp_blend = mlp_blend
        self.gbdt_blend = gbdt_blend
        self.predict_batch_size = predict_batch_size
        self.embed_batch_size = embed_batch_size
        self.seed = seed

        self.gbdt = None  # will be fitted in phase 2

    def _prep_x(self, X: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(X.reshape(len(X), -1).astype(np.float32))
        x = x / 255.0
        x = (x - 0.1307) / 0.3081
        return x

    @torch.no_grad()
    def _embed(self, X: torch.Tensor) -> np.ndarray:
        """Extract 256-dim features from trained backbone."""
        self.model.eval()
        feats = []
        bs = self.embed_batch_size
        for i in range(0, X.shape[0], bs):
            feats.append(self.model.embed(X[i : i + bs]).cpu().numpy())
        return np.concatenate(feats, axis=0)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None):
        t0 = time.perf_counter()
        total_budget = self.episode_budget_s - self.safety_margin_s
        mlp_deadline = t0 + total_budget * self.mlp_frac
        hard_deadline = t0 + total_budget

        X = self._prep_x(X_train).to(self.device)
        y_int = y_train.ravel().astype(np.int64)
        y = torch.from_numpy(y_int).to(self.device)
        n = X.shape[0]

        # ------------------------------------------------------------------
        # Phase 1: MLP training
        # ------------------------------------------------------------------
        self.model.train()
        epoch = 0
        while time.perf_counter() < mlp_deadline:
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                if time.perf_counter() >= mlp_deadline:
                    break
                idx = perm[i : i + self.batch_size]
                logits = self.model(X[idx])
                loss = self.criterion(logits, y[idx])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step(epoch)
            epoch += 1

        # ------------------------------------------------------------------
        # Phase 2: GBDT on neural features (if sklearn available + time left)
        # ------------------------------------------------------------------
        if not _SKLEARN_AVAILABLE:
            return

        time_left = hard_deadline - time.perf_counter()
        if time_left < 2.0:
            return

        # Extract embeddings for all training samples
        train_feats = self._embed(X)  # (60000, 256)

        # Fit HistGradientBoosting with remaining time as a rough guide
        # Use random state for reproducibility
        rs = self.seed if self.seed is not None else 0
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=self.gbdt_max_iter,
            max_leaf_nodes=self.gbdt_max_leaf_nodes,
            learning_rate=self.gbdt_learning_rate,
            random_state=rs,
            verbose=0,
        )
        self.gbdt.fit(train_feats, y_int)

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = self._prep_x(X_test).to(self.device)
        self.model.eval()
        bs = self.predict_batch_size

        # MLP softmax probs
        mlp_probs_list = []
        for i in range(0, X.shape[0], bs):
            mlp_probs_list.append(torch.softmax(self.model(X[i : i + bs]), dim=1).cpu())
        mlp_probs = torch.cat(mlp_probs_list, dim=0).numpy()  # (10000, 10)

        if self.gbdt is None or not _SKLEARN_AVAILABLE:
            return mlp_probs.argmax(axis=1)

        # GBDT probs on neural features
        test_feats = self._embed(X)  # (10000, 256)
        gbdt_probs = self.gbdt.predict_proba(test_feats)  # (10000, 10)

        # Blend
        blended = self.mlp_blend * mlp_probs + self.gbdt_blend * gbdt_probs
        return blended.argmax(axis=1)
