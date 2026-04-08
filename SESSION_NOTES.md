# Session Notes — Permuted MNIST Competition

## Goal

Maximise classification accuracy on the [Permuted MNIST competition](https://ml-arena.com/viewcompetition/8).
Target: 99%+. Constraints: 60 s wall-clock per episode, 2 CPU cores, 4 GB RAM, no GPU.
Evaluation runs inside a Podman container via `python eval.py`.

---

## Key Constraints & Discoveries

### Timing reality
- A single epoch over 60 k samples at `batch_size=512` takes **~2.4–3.1 s** on 2 CPU cores.
- With a 57.5 s training budget only **~20–22 epochs** actually fit — not 180+ as previously assumed.
- This reframes the problem: fast convergence in ~20 epochs, not exploiting many epochs.

### Data characteristics
- After permutation + eval-time noise: global mean ≈ 0.135, std ≈ 0.304 (close to standard MNIST 0.1307 / 0.3081).
- Eval environment adds mild per-image Gaussian noise (std ≈ 0.015 in pixel space ≈ 0.049 in normalised space), brightness ±4%, shift ±2%.

### Architecture speed
| Architecture | s/epoch | Epochs in 57 s |
|---|---|---|
| 784→512→512→256→10 | ~2.6 s | ~22 |
| 784→1024→1024→512→10 | ~7.8 s | ~7 |
| 784→384→384→192→10 | ~2.1 s | ~27 |

The standard 512-width 3-layer MLP is the best trade-off.

---

## Sweep Results

### Sweep 1 — baseline hyperparameter grid
Variables: `label_smoothing`, `lr`, `batch_size`, `weight_decay`, `scheduler`.

Key findings:
- `label_smoothing=0` consistently **worse** than 0.05–0.10.
- `lr=3e-3` with `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` is near-optimal.
- `batch_size=512` beats 256 and 1024 for this architecture.
- Best single-episode: `ls=0.10, lr=5e-3` → 98.74% (same as baseline).

### Sweep 2 — architecture & augmentation
Variables: wider/deeper/narrower MLP, Gaussian noise augmentation, SWA, 2-model ensemble.

Key findings:
- **Gaussian noise augmentation is the breakthrough**: `aug_noise=0.10` → 98.82% (vs 98.74% baseline).
- SWA: 98.72% — no improvement.
- 2-model ensemble (split budget): 98.76% — marginal improvement.
- Wider/deeper networks slower, fewer epochs, worse result.

### Sweep 3 — fine-tune noise level
Variables: noise in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20], label_smoothing, lr, dropout, weight_decay.

Key findings:
- `noise=0.12` → 98.89% (single episode).
- `noise=0.20` → 98.89% (single episode).
- Noise keeps helping as it increases — not plateauing yet.

### Sweep 4 — multi-episode validation (5 episodes)
Validated top candidates from sweep 3.

| Config | Mean (5 ep) |
|---|---|
| noise=0.30, ls=0.10 | 98.88% |
| noise=0.25, ls=0.10 | 98.84% |
| noise=0.20, ls=0.10 | 98.84% |
| Baseline (no noise) | 98.72% |

### Sweep 5 — push noise higher + ensemble×noise
Variables: noise in [0.30, 0.40, 0.50, 0.60], ensemble×2 with noise.

| Config | Mean (5 ep) |
|---|---|
| **noise=0.50, ls=0.10** | **98.97%** |
| noise=0.60, ls=0.10 | 98.93% |
| noise=0.40, ls=0.10 | 98.91% |
| ensemble×2, noise=0.30 | 98.53% (high variance) |

`noise=0.50` is the sweet spot. Ensemble with split budget hurts (fewer epochs per model).

### Sweep 6 — fine-tune around noise=0.50 (5 episodes each)
Variables: noise, ls, lr, wd, dropout.

| Config | Mean (5 ep) |
|---|---|
| noise=0.50, ls=0.10, dropout=0.00 | 98.97% |
| noise=0.70, ls=0.10 | 98.95% |
| noise=0.50, ls=0.15 | 98.96% |
| noise=0.50, wd=0 | 98.96% |

Key finding: with `aug_noise=0.50`, **dropout can be removed** (noise is sufficient regularisation).

### Sweep 7 — full 10-episode validation

| Config | Mean (10 ep) | Std |
|---|---|---|
| noise=0.50, ls=0.10, do=0.00 | 98.96% | 0.0003 |
| noise=0.50, ls=0.10, do=0.02 | 98.92% | 0.0007 |
| noise=0.50, ls=0.15, do=0.05 | 98.91% | 0.0005 |
| noise=0.70, ls=0.10, do=0.05 | 98.88% | 0.0012 |

---

## Approaches That Did NOT Help

| Approach | Result |
|---|---|
| Pseudo-label self-training | 98.67–98.73% |
| kNN ensemble for uncertain samples | 98.27% |
| Lion optimizer + OneCycleLR | 98.37% |
| Deep ensemble (2× MLP, split budget) | 98.72% |
| TTA (8–12 augmented passes) | 98.70–98.72% |
| MLP + GBDT on neural features | 98.47% |
| Larger networks (1024→1024→512) | 98.32% |
| Residual blocks | 98.71% |
| Mixup augmentation | 98.56% |
| MC-Dropout at inference | 98.69% |
| SWA (Stochastic Weight Averaging) | 98.72% |
| Per-pixel z-score normalisation | 98.14% (worse than global) |

---

## Final agent.py Configuration

```
Architecture:  784 → 512 → 512 → 256 → 10 (BatchNorm + GELU after each hidden layer)
Dropout:       0.0  (noise provides regularisation)
Optimiser:     AdamW  lr=3e-3  weight_decay=1e-4
Scheduler:     CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-5)
Loss:          CrossEntropyLoss(label_smoothing=0.10)
Augmentation:  xb += randn_like(xb) * 0.50  (every mini-batch during training)
Batch size:    512
Training time: ~57.6 s per episode  (safety_margin_s=2.5)
Normalisation: (x/255 − 0.1307) / 0.3081
```

### Official eval results (10 episodes, `python eval.py` in container)

```
Episode  1: 0.9889   Episode  6: 0.9892
Episode  2: 0.9887   Episode  7: 0.9902
Episode  3: 0.9891   Episode  8: 0.9893
Episode  4: 0.9892   Episode  9: 0.9887
Episode  5: 0.9898   Episode 10: 0.9894

Mean: 0.9892  (+/- 0.0004)   Time: 57.6 s/episode
```

**Improvement over baseline: +0.18 pp (98.74% → 98.92%)**

---

## Why Noise Augmentation Works

The eval environment adds mild Gaussian noise to each test image. Training with amplified noise (`std=0.50` in normalised space ≈ `std=0.154` in pixel space, ~10× the eval noise) forces the model to learn features that are robust to noise perturbations. This acts as both a data-augmentation strategy and an implicit regulariser, replacing the need for explicit dropout.

---

## Remaining Gap to 99%

~0.08 pp. Approaches exhausted. The residual error appears to be a combination of:
1. Irreducible label noise in the permuted+augmented data.
2. Capacity limitations of a 20-epoch training regime on CPU.
3. The permutation destroying spatial structure that CNNs would exploit.
