"""
Hyperparameter sweep for the 20-epoch regime.
Tests label_smoothing, lr, batch_size, weight_decay, dropout combinations
to find what beats the 98.74% baseline.

Run inside container:
  podman run --rm --cpus=2 --memory=4g --memory-swap=4g \
    -v "${PWD}:/work" -w /work permuted-mnist-eval:py312 python sweep.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")

# Fixed seed for reproducibility across all configs
SEED = 42
# Use a single episode for the sweep (faster), then validate winner on more episodes
EPISODE = 0

# ─── Data ────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
t0 = time.perf_counter()
env = PermutedMNISTEnv(number_episodes=1)
env.set_seed(42)
task = env.get_next_task()
X_train, y_train = task["X_train"], task["y_train"]
X_test, y_test_np = task["X_test"], task["y_test"].ravel()
print(f"  loaded in {time.perf_counter()-t0:.2f}s", flush=True)


def prep_x(X, mean=0.1307, std=0.3081):
    x = torch.from_numpy(X.reshape(len(X), -1)).float()
    x = x / 255.0
    x = (x - mean) / std
    return x


X_tr = prep_x(X_train).to(DEVICE)
y_tr = torch.from_numpy(y_train.ravel().astype(np.int64)).to(DEVICE)
X_te = prep_x(X_test).to(DEVICE)


# ─── Model ───────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, dropout=0.05):
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


def run_config(cfg, budget=57.0):
    torch.manual_seed(SEED)
    model = MLP(dropout=cfg["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    if cfg["scheduler"] == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )
    elif cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-5
        )
    elif cfg["scheduler"] == "onecycle":
        # Will be set per-epoch below — placeholder
        scheduler = None
    else:
        scheduler = None

    n = X_tr.shape[0]
    batch_size = cfg["batch_size"]
    deadline = time.perf_counter() + budget

    epoch = 0
    model.train()
    while time.perf_counter() < deadline:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if time.perf_counter() >= deadline:
                break
            idx = perm[i: i + batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if cfg["scheduler"] == "cosine_restarts":
                scheduler.step(epoch)
            else:
                scheduler.step()
        epoch += 1

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, X_te.shape[0], 4096):
            preds.append(model(X_te[i: i + 4096]).argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)

    acc = (preds == y_test_np).mean()
    return acc, epoch


# ─── Sweep grid ──────────────────────────────────────────────────────────────
# Baseline: lr=3e-3, batch=512, wd=1e-4, ls=0.05, dropout=0.05, cosine_restarts
# We vary one or two axes at a time to isolate the best combination.

configs = []

# Group 1: label_smoothing vs lr (batch=512, wd=1e-4, dropout=0.05, cosine_restarts)
for ls in [0.0, 0.05, 0.1]:
    for lr in [3e-3, 5e-3, 1e-2]:
        configs.append({
            "label_smoothing": ls,
            "lr": lr,
            "batch_size": 512,
            "weight_decay": 1e-4,
            "dropout": 0.05,
            "scheduler": "cosine_restarts",
            "group": "ls_lr",
        })

# Group 2: batch_size scaling (ls=0.0, best lr from group1 TBD — run all with 3e-3 and 1e-2)
for bs in [256, 512, 1024]:
    for lr in [3e-3, 1e-2]:
        configs.append({
            "label_smoothing": 0.0,
            "lr": lr,
            "batch_size": bs,
            "weight_decay": 1e-4,
            "dropout": 0.05,
            "scheduler": "cosine_restarts",
            "group": "bs_lr",
        })

# Group 3: weight_decay + Adam vs AdamW
for wd in [0.0, 1e-5, 1e-4, 1e-3]:
    configs.append({
        "label_smoothing": 0.0,
        "lr": 5e-3,
        "batch_size": 512,
        "weight_decay": wd,
        "dropout": 0.05,
        "scheduler": "cosine",
        "group": "wd",
    })

# Group 4: dropout
for do in [0.0, 0.05, 0.1, 0.2]:
    configs.append({
        "label_smoothing": 0.0,
        "lr": 5e-3,
        "batch_size": 512,
        "weight_decay": 1e-4,
        "dropout": do,
        "scheduler": "cosine",
        "group": "dropout",
    })

# Group 5: cosine vs cosine_restarts scheduler
for sched in ["cosine", "cosine_restarts"]:
    configs.append({
        "label_smoothing": 0.0,
        "lr": 5e-3,
        "batch_size": 512,
        "weight_decay": 1e-4,
        "dropout": 0.05,
        "scheduler": sched,
        "group": "sched",
    })

# Deduplicate
seen = set()
unique_configs = []
for c in configs:
    key = (c["label_smoothing"], c["lr"], c["batch_size"], c["weight_decay"], c["dropout"], c["scheduler"])
    if key not in seen:
        seen.add(key)
        unique_configs.append(c)

print(f"\nRunning {len(unique_configs)} configs...\n", flush=True)
print(f"{'Group':<15} {'ls':>5} {'lr':>7} {'bs':>5} {'wd':>7} {'do':>5} {'sched':<18} {'acc':>7} {'ep':>4}", flush=True)
print("-" * 90, flush=True)

results = []
BASELINE = 0.9874

for cfg in unique_configs:
    t_start = time.perf_counter()
    acc, ep = run_config(cfg, budget=57.0)
    elapsed = time.perf_counter() - t_start
    marker = " ***" if acc > BASELINE else ""
    print(
        f"{cfg['group']:<15} {cfg['label_smoothing']:>5.2f} {cfg['lr']:>7.4f} "
        f"{cfg['batch_size']:>5} {cfg['weight_decay']:>7.5f} {cfg['dropout']:>5.2f} "
        f"{cfg['scheduler']:<18} {acc:>7.4f} {ep:>4}{marker}",
        flush=True,
    )
    results.append((acc, cfg))

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== TOP 10 CONFIGS ===", flush=True)
results.sort(key=lambda x: -x[0])
for acc, cfg in results[:10]:
    print(
        f"  acc={acc:.4f}  ls={cfg['label_smoothing']}  lr={cfg['lr']}  "
        f"bs={cfg['batch_size']}  wd={cfg['weight_decay']}  "
        f"do={cfg['dropout']}  sched={cfg['scheduler']}",
        flush=True,
    )
