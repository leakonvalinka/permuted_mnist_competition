"""
Final validation: Run best config on all 10 episodes.
Best config: noise=0.50, ls=0.10, dropout=0.0, lr=3e-3, bs=512, wd=1e-4

Run inside container:
  podman run --rm --cpus=2 --memory=4g --memory-swap=4g \
    -v "${PWD}:/work" -w /work permuted-mnist-eval:py312 python sweep7_final.py
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
N_EPISODES = 10


class MLP(nn.Module):
    def __init__(self, dropout=0.0):
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
    model = MLP(dropout=cfg.get("dropout", 0.0)).to(DEVICE)
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


# The best configs to validate with full 10 episodes
candidates = [
    ("noise=0.50 ls=0.10 do=0.00",  dict(noise=0.50, ls=0.10, dropout=0.00)),
    ("noise=0.50 ls=0.15 do=0.05",  dict(noise=0.50, ls=0.15, dropout=0.05)),
    ("noise=0.70 ls=0.10 do=0.05",  dict(noise=0.70, ls=0.10, dropout=0.05)),
    ("noise=0.50 ls=0.10 do=0.02",  dict(noise=0.50, ls=0.10, dropout=0.02)),
]

print(f"Running {len(candidates)} candidates × {N_EPISODES} episodes each...\n", flush=True)
BASELINE = 0.9874

all_results = []
for name, cfg in candidates:
    env = PermutedMNISTEnv(number_episodes=N_EPISODES)
    env.set_seed(42)
    accs = []
    times = []
    for _ in range(N_EPISODES):
        task = env.get_next_task()
        t0 = time.perf_counter()
        acc = run_episode(task["X_train"], task["y_train"], task["X_test"], task["y_test"].ravel(), cfg)
        times.append(time.perf_counter() - t0)
        accs.append(acc)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_t = np.mean(times)
    marker = " ***" if mean_acc > BASELINE else ""
    print(
        f"{name:<45}  mean={mean_acc:.4f} std={std_acc:.4f}  t={mean_t:.1f}s",
        flush=True,
    )
    print(f"  per-ep: {[f'{a:.4f}' for a in accs]}{marker}", flush=True)
    all_results.append((mean_acc, std_acc, name))

print("\n=== RANKED (10 episodes) ===", flush=True)
all_results.sort(key=lambda x: -x[0])
for mean_acc, std_acc, name in all_results:
    marker = " ***" if mean_acc > BASELINE else ""
    print(f"  mean={mean_acc:.4f} std={std_acc:.4f}  {name}{marker}", flush=True)
