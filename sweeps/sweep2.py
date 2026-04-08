"""
Targeted sweep round 2:
- Try augmentation during training (Gaussian noise matching eval noise)
- Try deeper model architectures that still fit many epochs
- Try ensemble of 2 models trained with different seeds
- Try SWA (Stochastic Weight Averaging)

Run inside container:
  podman run --rm --cpus=2 --memory=4g --memory-swap=4g \
    -v "${PWD}:/work" -w /work permuted-mnist-eval:py312 python sweep2.py
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

# ─── Data ────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
env = PermutedMNISTEnv(number_episodes=1)
env.set_seed(42)
task = env.get_next_task()
X_train, y_train = task["X_train"], task["y_train"]
X_test, y_test_np = task["X_test"], task["y_test"].ravel()


def prep_x(X):
    x = torch.from_numpy(X.reshape(len(X), -1)).float()
    x = x / 255.0
    x = (x - 0.1307) / 0.3081
    return x


X_tr = prep_x(X_train).to(DEVICE)
y_tr = torch.from_numpy(y_train.ravel().astype(np.int64)).to(DEVICE)
X_te = prep_x(X_test).to(DEVICE)
n = X_tr.shape[0]


def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, X_te.shape[0], 4096):
            preds.append(model(X_te[i: i + 4096]).argmax(dim=1).cpu().numpy())
    return (np.concatenate(preds) == y_test_np).mean()


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Standard 784→512→512→256→10"""
    def __init__(self, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 10),
        )
    def forward(self, x): return self.net(x)


class MLP_Wide(nn.Module):
    """784→768→768→384→10 — slightly wider"""
    def __init__(self, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(768, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 10),
        )
    def forward(self, x): return self.net(x)


class MLP_Deep(nn.Module):
    """784→512→512→512→256→10 — one extra layer"""
    def __init__(self, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 10),
        )
    def forward(self, x): return self.net(x)


class MLP_Narrow(nn.Module):
    """784→384→384→192→10 — narrower but faster (more epochs)"""
    def __init__(self, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 10),
        )
    def forward(self, x): return self.net(x)


# ─── Training helpers ─────────────────────────────────────────────────────────
def make_optimizer_scheduler(model, lr=3e-3, wd=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    return opt, sched


def train_model(model, aug_noise=0.0, budget=BUDGET, lr=3e-3, wd=1e-4,
                label_smoothing=0.05, batch_size=512):
    torch.manual_seed(SEED)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt, sched = make_optimizer_scheduler(model, lr=lr, wd=wd)
    deadline = time.perf_counter() + budget

    epoch = 0
    model.train()
    while time.perf_counter() < deadline:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if time.perf_counter() >= deadline:
                break
            idx = perm[i: i + batch_size]
            xb = X_tr[idx]
            if aug_noise > 0.0:
                xb = xb + torch.randn_like(xb) * aug_noise
            logits = model(xb)
            loss = criterion(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step(epoch)
        epoch += 1

    return epoch


def train_with_swa(model, budget=BUDGET, lr=3e-3, wd=1e-4,
                   label_smoothing=0.05, batch_size=512, swa_start_frac=0.75):
    """Train with SWA applied in the last fraction of training."""
    torch.manual_seed(SEED)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt, sched = make_optimizer_scheduler(model, lr=lr, wd=wd)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(opt, swa_lr=5e-4)

    deadline = time.perf_counter() + budget
    swa_deadline = time.perf_counter() + budget * swa_start_frac

    epoch = 0
    in_swa = False
    model.train()
    while time.perf_counter() < deadline:
        # Switch to SWA phase
        if (not in_swa) and time.perf_counter() >= swa_deadline:
            in_swa = True

        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if time.perf_counter() >= deadline:
                break
            idx = perm[i: i + batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            sched.step(epoch)
        epoch += 1

    # Update BN stats for SWA model
    swa_model.train()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            swa_model(X_tr[i: i + batch_size])
    return swa_model, epoch


def train_ensemble_2(budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.05, batch_size=512):
    """Train 2 models on split time budget and average logits."""
    half = budget / 2.0
    models = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        m = MLP(dropout=0.05).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        opt, sched = make_optimizer_scheduler(m, lr=lr, wd=wd)
        deadline = time.perf_counter() + half
        epoch = 0
        m.train()
        while time.perf_counter() < deadline:
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                if time.perf_counter() >= deadline:
                    break
                idx = perm[i: i + batch_size]
                logits = m(X_tr[idx])
                loss = criterion(logits, y_tr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step(epoch)
            epoch += 1
        models.append(m)
    return models


def eval_ensemble(models):
    for m in models:
        m.eval()
    with torch.no_grad():
        all_logits = []
        for i in range(0, X_te.shape[0], 4096):
            xb = X_te[i: i + 4096]
            logits = sum(m(xb) for m in models) / len(models)
            all_logits.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_logits)
    return (preds == y_test_np).mean()


# ─── Experiments ─────────────────────────────────────────────────────────────
results = []
BASELINE = 0.9874

def run(name, fn):
    t0 = time.perf_counter()
    acc = fn()
    elapsed = time.perf_counter() - t0
    marker = " ***" if acc > BASELINE else ""
    print(f"{name:<50} acc={acc:.4f}  t={elapsed:.1f}s{marker}", flush=True)
    results.append((acc, name))
    return acc


print("\n--- Architecture sweep ---", flush=True)

def exp_baseline():
    m = MLP(0.05).to(DEVICE)
    train_model(m, aug_noise=0.0, budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.05)
    return evaluate(m)

def exp_ls010_lr5e3():
    m = MLP(0.05).to(DEVICE)
    train_model(m, aug_noise=0.0, budget=BUDGET, lr=5e-3, wd=1e-4, label_smoothing=0.1)
    return evaluate(m)

def exp_wide():
    m = MLP_Wide(0.05).to(DEVICE)
    train_model(m, aug_noise=0.0, budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.1)
    return evaluate(m)

def exp_deep():
    m = MLP_Deep(0.05).to(DEVICE)
    train_model(m, aug_noise=0.0, budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.1)
    return evaluate(m)

def exp_narrow():
    m = MLP_Narrow(0.05).to(DEVICE)
    train_model(m, aug_noise=0.0, budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.1)
    return evaluate(m)

run("Baseline (ls=0.05, lr=3e-3)", exp_baseline)
run("ls=0.10 lr=5e-3 (best sweep1)", exp_ls010_lr5e3)
run("MLP_Wide 768→768→384 ls=0.1", exp_wide)
run("MLP_Deep 512→512→512→256 ls=0.1", exp_deep)
run("MLP_Narrow 384→384→192 ls=0.1", exp_narrow)


print("\n--- Augmentation sweep (noise added to inputs) ---", flush=True)

for noise in [0.03, 0.05, 0.10, 0.15]:
    def exp_noise(n=noise):
        m = MLP(0.05).to(DEVICE)
        train_model(m, aug_noise=n, budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.1)
        return evaluate(m)
    run(f"aug_noise={noise}", exp_noise)


print("\n--- SWA (Stochastic Weight Averaging) ---", flush=True)

def exp_swa():
    m = MLP(0.05).to(DEVICE)
    swa_model, ep = train_with_swa(m, budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.1)
    swa_model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, X_te.shape[0], 4096):
            preds.append(swa_model(X_te[i: i + 4096]).argmax(dim=1).cpu().numpy())
    return (np.concatenate(preds) == y_test_np).mean()

run("SWA (last 25% of training)", exp_swa)


print("\n--- 2-model ensemble (split budget) ---", flush=True)

def exp_ensemble():
    models = train_ensemble_2(budget=BUDGET, lr=3e-3, wd=1e-4, label_smoothing=0.1)
    return eval_ensemble(models)

run("2-model ensemble (split 57s)", exp_ensemble)


print("\n=== SUMMARY ===", flush=True)
results.sort(key=lambda x: -x[0])
for acc, name in results:
    marker = " ***" if acc > BASELINE else ""
    print(f"  {acc:.4f}  {name}{marker}", flush=True)
