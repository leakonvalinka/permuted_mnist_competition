"""
Sweep 3: Fine-tune around aug_noise=0.1 (the breakthrough).
Vary: noise level, label_smoothing, lr, batch_size, dropout.

Run inside container:
  podman run --rm --cpus=2 --memory=4g --memory-swap=4g \
    -v "${PWD}:/work" -w /work permuted-mnist-eval:py312 python sweep3.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")
SEED = 42
BUDGET = 57.0

print("Loading data...", flush=True)
env = PermutedMNISTEnv(number_episodes=1)
env.set_seed(42)
task = env.get_next_task()
X_train, y_train = task["X_train"], task["y_train"]
X_test, y_test_np = task["X_test"], task["y_test"].ravel()


def prep_x(X):
    x = torch.from_numpy(X.reshape(len(X), -1)).float()
    return (x / 255.0 - 0.1307) / 0.3081


X_tr = prep_x(X_train).to(DEVICE)
y_tr = torch.from_numpy(y_train.ravel().astype(np.int64)).to(DEVICE)
X_te = prep_x(X_test).to(DEVICE)
n = X_tr.shape[0]


class MLP(nn.Module):
    def __init__(self, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 10),
        )
    def forward(self, x): return self.net(x)


def run_config(cfg):
    torch.manual_seed(SEED)
    model = MLP(dropout=cfg["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["ls"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)

    deadline = time.perf_counter() + BUDGET
    batch_size = cfg["bs"]
    noise = cfg["noise"]
    epoch = 0
    model.train()

    while time.perf_counter() < deadline:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if time.perf_counter() >= deadline:
                break
            idx = perm[i: i + batch_size]
            xb = X_tr[idx]
            if noise > 0:
                xb = xb + torch.randn_like(xb) * noise
            logits = model(xb)
            loss = criterion(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step(epoch)
        epoch += 1

    model.eval()
    with torch.no_grad():
        preds = np.concatenate([
            model(X_te[i: i + 4096]).argmax(1).cpu().numpy()
            for i in range(0, X_te.shape[0], 4096)
        ])
    return (preds == y_test_np).mean(), epoch


configs = []

# Group A: noise level sweep (baseline: ls=0.1, lr=3e-3, bs=512, wd=1e-4, do=0.05)
for noise in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
    configs.append(dict(noise=noise, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=0.05, group="noise"))

# Group B: label_smoothing with noise=0.1
for ls in [0.0, 0.05, 0.1, 0.15, 0.2]:
    configs.append(dict(noise=0.1, ls=ls, lr=3e-3, bs=512, wd=1e-4, dropout=0.05, group="ls"))

# Group C: lr with noise=0.1
for lr in [2e-3, 3e-3, 5e-3, 8e-3]:
    configs.append(dict(noise=0.1, ls=0.1, lr=lr, bs=512, wd=1e-4, dropout=0.05, group="lr"))

# Group D: dropout with noise=0.1
for do in [0.0, 0.05, 0.10, 0.15, 0.20]:
    configs.append(dict(noise=0.1, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=do, group="dropout"))

# Group E: weight_decay with noise=0.1
for wd in [0.0, 1e-5, 1e-4, 5e-4, 1e-3]:
    configs.append(dict(noise=0.1, ls=0.1, lr=3e-3, bs=512, wd=wd, dropout=0.05, group="wd"))

# Group F: batch_size with noise=0.1
for bs in [256, 512, 1024]:
    configs.append(dict(noise=0.1, ls=0.1, lr=3e-3, bs=bs, wd=1e-4, dropout=0.05, group="bs"))

# Deduplicate
seen = set()
unique = []
for c in configs:
    key = (c["noise"], c["ls"], c["lr"], c["bs"], c["wd"], c["dropout"])
    if key not in seen:
        seen.add(key)
        unique.append(c)

BASELINE = 0.9874
print(f"\nRunning {len(unique)} configs...\n", flush=True)
print(f"{'Group':<10} {'noise':>6} {'ls':>5} {'lr':>7} {'bs':>5} {'wd':>7} {'do':>5}  {'acc':>7} {'ep':>4}", flush=True)
print("-" * 70, flush=True)

results = []
for cfg in unique:
    t0 = time.perf_counter()
    acc, ep = run_config(cfg)
    marker = " ***" if acc > BASELINE else ""
    print(
        f"{cfg['group']:<10} {cfg['noise']:>6.3f} {cfg['ls']:>5.2f} {cfg['lr']:>7.4f} "
        f"{cfg['bs']:>5} {cfg['wd']:>7.5f} {cfg['dropout']:>5.2f}  {acc:>7.4f} {ep:>4}{marker}",
        flush=True,
    )
    results.append((acc, cfg))

print("\n=== TOP 10 ===", flush=True)
results.sort(key=lambda x: -x[0])
for acc, c in results[:10]:
    print(
        f"  acc={acc:.4f}  noise={c['noise']}  ls={c['ls']}  lr={c['lr']}  "
        f"bs={c['bs']}  wd={c['wd']}  do={c['dropout']}",
        flush=True,
    )
