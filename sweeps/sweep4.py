"""
Sweep 4: Validate top candidates across 5 episodes.
Best single-episode: noise=0.12 / noise=0.20, ls=0.1, lr=3e-3, bs=512, wd=1e-4, do=0.05

Also try a few combinations of noise+ls+wd that look promising.

Run inside container:
  podman run --rm --cpus=2 --memory=4g --memory-swap=4g \
    -v "${PWD}:/work" -w /work permuted-mnist-eval:py312 python sweep4.py
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
    model = MLP(dropout=cfg["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["ls"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)

    X_tr = prep_x(X_train).to(DEVICE)
    y_tr = torch.from_numpy(y_train.ravel().astype(np.int64)).to(DEVICE)
    X_te = prep_x(X_test).to(DEVICE)
    n = X_tr.shape[0]

    deadline = time.perf_counter() + BUDGET
    bs = cfg["bs"]
    noise = cfg["noise"]
    epoch = 0
    model.train()

    while time.perf_counter() < deadline:
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if time.perf_counter() >= deadline:
                break
            idx = perm[i: i + bs]
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
    return (preds == y_test_np).mean()


# Candidates to evaluate
candidates = [
    # Baseline
    dict(name="BASELINE (no noise, ls=0.05, lr=3e-3)",
         noise=0.0, ls=0.05, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
    # Top from sweep3 single episode
    dict(name="noise=0.12 ls=0.10",
         noise=0.12, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
    dict(name="noise=0.20 ls=0.10",
         noise=0.20, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
    dict(name="noise=0.10 ls=0.10 wd=1e-3",
         noise=0.10, ls=0.1, lr=3e-3, bs=512, wd=1e-3, dropout=0.05),
    dict(name="noise=0.15 ls=0.10",
         noise=0.15, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
    # Combo: medium noise + high label smoothing
    dict(name="noise=0.12 ls=0.15 wd=1e-3",
         noise=0.12, ls=0.15, lr=3e-3, bs=512, wd=1e-3, dropout=0.05),
    # Combo: high noise + standard ls
    dict(name="noise=0.20 ls=0.05",
         noise=0.20, ls=0.05, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
    # Larger noise
    dict(name="noise=0.25 ls=0.10",
         noise=0.25, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
    dict(name="noise=0.30 ls=0.10",
         noise=0.30, ls=0.1, lr=3e-3, bs=512, wd=1e-4, dropout=0.05),
]

print(f"Running {len(candidates)} candidates × {N_EPISODES} episodes each...\n", flush=True)
BASELINE_MEAN = 0.9874

all_results = []
for cfg in candidates:
    env = PermutedMNISTEnv(number_episodes=N_EPISODES)
    env.set_seed(42)
    accs = []
    for ep_idx in range(N_EPISODES):
        task = env.get_next_task()
        acc = run_episode(task["X_train"], task["y_train"], task["X_test"], task["y_test"].ravel(), cfg)
        accs.append(acc)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    marker = " ***" if mean_acc > BASELINE_MEAN else ""
    print(
        f"{cfg['name']:<45}  mean={mean_acc:.4f} std={std_acc:.4f}  {[f'{a:.4f}' for a in accs]}{marker}",
        flush=True,
    )
    all_results.append((mean_acc, std_acc, cfg))

print("\n=== RANKED ===", flush=True)
all_results.sort(key=lambda x: -x[0])
for mean_acc, std_acc, cfg in all_results:
    marker = " ***" if mean_acc > BASELINE_MEAN else ""
    print(f"  mean={mean_acc:.4f} std={std_acc:.4f}  {cfg['name']}{marker}", flush=True)
