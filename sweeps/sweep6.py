"""
Sweep 6: Fine-tune around noise=0.50 ls=0.10 (98.97% mean, 5ep).
Vary: noise, ls, lr, wd, dropout.

Run inside container:
  podman run --rm --cpus=2 --memory=4g --memory-swap=4g \
    -v "${PWD}:/work" -w /work permuted-mnist-eval:py312 python sweep6.py
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
N_EPISODES = 5


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


def prep_x(X):
    x = torch.from_numpy(X.reshape(len(X), -1)).float()
    return (x / 255.0 - 0.1307) / 0.3081


def run_episode(X_train, y_train, X_test, y_test_np, cfg):
    torch.manual_seed(SEED)
    model = MLP(dropout=cfg.get("dropout", 0.05)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["ls"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-3), weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)

    X_tr = prep_x(X_train).to(DEVICE)
    y_tr = torch.from_numpy(y_train.ravel().astype(np.int64)).to(DEVICE)
    X_te = prep_x(X_test).to(DEVICE)
    n = X_tr.shape[0]

    deadline = time.perf_counter() + BUDGET
    bs = cfg.get("bs", 512)
    noise = cfg["noise"]
    epoch = 0
    model.train()

    while time.perf_counter() < deadline:
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if time.perf_counter() >= deadline:
                break
            idx = perm[i: i + bs]
            xb = X_tr[idx] + torch.randn_like(X_tr[idx]) * noise
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
    return (preds == y_test_np).mean()


def run_candidate(name, cfg):
    env = PermutedMNISTEnv(number_episodes=N_EPISODES)
    env.set_seed(42)
    accs = []
    for _ in range(N_EPISODES):
        task = env.get_next_task()
        acc = run_episode(task["X_train"], task["y_train"], task["X_test"], task["y_test"].ravel(), cfg)
        accs.append(acc)
    return np.mean(accs), np.std(accs), accs


BASELINE = 0.9897  # noise=0.50 result from sweep5
ORIG_BASELINE = 0.9874

candidates = [
    # noise sweep around 0.50
    ("noise=0.45 ls=0.10",       dict(noise=0.45, ls=0.10)),
    ("noise=0.50 ls=0.10 [ref]", dict(noise=0.50, ls=0.10)),
    ("noise=0.55 ls=0.10",       dict(noise=0.55, ls=0.10)),
    ("noise=0.70 ls=0.10",       dict(noise=0.70, ls=0.10)),
    ("noise=1.00 ls=0.10",       dict(noise=1.00, ls=0.10)),
    # label smoothing with noise=0.50
    ("noise=0.50 ls=0.05",       dict(noise=0.50, ls=0.05)),
    ("noise=0.50 ls=0.15",       dict(noise=0.50, ls=0.15)),
    ("noise=0.50 ls=0.20",       dict(noise=0.50, ls=0.20)),
    # lr with noise=0.50
    ("noise=0.50 ls=0.10 lr=2e-3",  dict(noise=0.50, ls=0.10, lr=2e-3)),
    ("noise=0.50 ls=0.10 lr=5e-3",  dict(noise=0.50, ls=0.10, lr=5e-3)),
    # wd with noise=0.50
    ("noise=0.50 ls=0.10 wd=0",     dict(noise=0.50, ls=0.10, wd=0.0)),
    ("noise=0.50 ls=0.10 wd=1e-3",  dict(noise=0.50, ls=0.10, wd=1e-3)),
    # dropout=0 (noise is already a regularizer)
    ("noise=0.50 ls=0.10 do=0",     dict(noise=0.50, ls=0.10, dropout=0.0)),
    ("noise=0.50 ls=0.10 do=0.02",  dict(noise=0.50, ls=0.10, dropout=0.02)),
]

print(f"Running {len(candidates)} candidates × {N_EPISODES} episodes each...\n", flush=True)
all_results = []

for name, cfg in candidates:
    mean_acc, std_acc, accs = run_candidate(name, cfg)
    marker = " ★★★" if mean_acc > BASELINE else (" ***" if mean_acc > ORIG_BASELINE else "")
    print(
        f"{name:<45}  mean={mean_acc:.4f} std={std_acc:.4f}  {[f'{a:.4f}' for a in accs]}{marker}",
        flush=True,
    )
    all_results.append((mean_acc, std_acc, name))

print("\n=== RANKED ===", flush=True)
all_results.sort(key=lambda x: -x[0])
for mean_acc, std_acc, name in all_results:
    marker = " ★★★" if mean_acc > BASELINE else (" ***" if mean_acc > ORIG_BASELINE else "")
    print(f"  mean={mean_acc:.4f} std={std_acc:.4f}  {name}{marker}", flush=True)
