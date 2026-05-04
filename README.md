<div align="center">

# Linear Regression — From Truth to Estimate

**OLS · Ridge · Lasso · on synthetic data with known ground-truth coefficients**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> Train three linear models on a synthetic dataset whose **true coefficients are known**, then ask the question that holdout RMSE alone can't answer:
> *did the model actually recover the truth?*

<table>
<tr>
<td align="center" width="33%">
<sub>Best test RMSE</sub><br>
<b style="font-size:1.6em; color:#3B6EA8;">0.992</b><br>
<sub>Lasso (α = 0.1)</sub>
</td>
<td align="center" width="33%">
<sub>Best test R²</sub><br>
<b style="font-size:1.6em; color:#3B6EA8;">0.829</b><br>
<sub>Lasso (α = 0.1)</sub>
</td>
<td align="center" width="33%">
<sub>Closest β recovery</sub><br>
<b style="font-size:1.6em; color:#C04040;">0.273</b><br>
<sub>‖β̂ − β_true‖₂ &nbsp; (Ridge)</sub>
</td>
</tr>
</table>

| Model | RMSE | MAE | R² | ‖β̂ − β_true‖₂ |
|---|---:|---:|---:|---:|
| OLS | 1.001 | 0.792 | 0.826 | 0.274 |
| **Ridge (α = 1.0)** | **1.000** | **0.792** | **0.826** | **0.273** ◀ best β recovery |
| **Lasso (α = 0.1)** | **0.992** | 0.797 | **0.829** | 0.537 ◀ best test fit |

<sub>**Headline finding:** with this much training data, all three models predict equally well — but they disagree on *how* they got there. Ridge stays closest to the true coefficients; Lasso wins on test fit by deliberately throwing away small ones.</sub>

---

## Dashboard

### 1. Test-set performance across models

![metrics](assets/01_metrics.png)

When you have enough data, regularized and unregularized linear regression all converge to roughly the same prediction quality. The interesting story is **not in this chart** — it's in the next one.

### 2. Coefficient recovery — the synthetic-data superpower

![coefficient recovery](assets/02_coef_recovery.png)

Light-gray bars are the **true** coefficients we used to generate `y`. Each colored bar is a model's estimate. The first 5 features (`x1`–`x5`) are *informative* (non-zero true β); the last 5 (`x6`–`x10`) are *redundant* (β = 0 by construction). Two things to read off:

- **OLS and Ridge** give nearly identical coefficient estimates — visually almost overlapping. Ridge's L2 shrinkage barely moves anything when the noise level is moderate.
- **Lasso** zeroes out *all five redundant features cleanly* (no red bars on `x6`–`x10`) but pays for it by under-shrinking the informative ones. That's the classical L1 trade-off: sparsity versus magnitude.

This kind of plot is **only possible because we generated the data ourselves** — there's no "true β" to plot for the California Housing dataset.

### 3. Predicted vs. actual on the test set

![pred vs actual](assets/03_pred_vs_actual.png)

The dashed red line is `y = ŷ` (perfect prediction). Points hug the line tightly across all three models — visual confirmation that test-set fit is essentially indistinguishable.

### 4. Residual diagnostics

![residuals](assets/04_residuals.png)

Residuals look like a structureless cloud around zero with no fan-out, no curvature, no clusters. That means the model isn't systematically wrong on any region of the input space — exactly what we want. The Gaussian noise we injected at generation time shows up here as the visible spread.

### 5. Lasso regularization path

![lasso path](assets/05_lasso_path.png)

Each line is one feature's coefficient, plotted as a function of the L1 strength α. Reading right-to-left (regularization getting stronger):

- **Blue lines (informative features)** persist far into the high-α region before being shrunk to zero.
- **Red lines (redundant features)** are killed off almost immediately as soon as α becomes non-trivial.

This is what people mean when they say "Lasso does feature selection." The path makes it visible.

---

## What's actually happening

### Ordinary Least Squares (OLS)

Find the coefficient vector β that minimizes the sum of squared residuals:

$$\hat{\beta}_{\text{OLS}} = \arg\min_{\beta} \\; \lVert y - X\beta \rVert_2^2$$

No bias, lowest variance among unbiased linear estimators (Gauss–Markov theorem) — but variance can still be huge when features are correlated. That's where regularization comes in.

### Ridge regression (L2)

$$\hat{\beta}_{\text{Ridge}} = \arg\min_{\beta} \\; \lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_2^2$$

Penalize large coefficients smoothly. Trades a little bias for less variance. Great when you have many correlated features (the squared penalty distributes weight across them rather than picking one arbitrarily).

### Lasso regression (L1)

$$\hat{\beta}_{\text{Lasso}} = \arg\min_{\beta} \\; \lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_1$$

Penalize the *absolute* sum of coefficients. The geometry of the L1 ball has corners, which is why Lasso pushes individual coefficients exactly to zero — it's not just shrinkage, it's selection.

### Key intuition

| Penalty | Geometry | Effect on β |
|---|---|---|
| None (OLS) | — | Unconstrained — fits noise as eagerly as signal |
| L2 (Ridge) | Smooth ball | Shrinks all coefficients toward zero, never to zero |
| L1 (Lasso) | Pointed diamond | Drives small coefficients exactly to zero — built-in feature selection |

---

## Reproduce

```bash
# 1. Set up the environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the synthetic dataset (deterministic given the seed)
python generate_data.py

# 3. Train, evaluate, and produce the dashboard figures
python train.py
```

After step 3, `assets/` will contain the five PNGs embedded above and `results/metrics.json` will hold the numeric summary.

### Tweak the difficulty

`DataConfig` in [`generate_data.py`](generate_data.py) exposes the knobs that change what story the dashboard tells:

```python
DataConfig(
    n_samples=1000,
    n_informative=5,    # how many features actually matter
    n_redundant=5,      # how many are pure noise — Lasso should drop these
    correlation=0.6,    # feature correlation — high values hurt OLS, help Ridge
    noise_std=1.0,      # observation noise level
    seed=42,
)
```

Try `correlation=0.95` or `n_samples=80` to see the regularizers visibly out-perform OLS.

---

## Project layout

```
01-linear-regression/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← synthetic dataset generator (deterministic)
├── train.py               ← OLS / Ridge / Lasso + figure pipeline
├── assets/                ← rendered dashboard figures
└── results/metrics.json   ← test-set metrics + β recovery error
```

---

## What I learned

- **Holdout RMSE alone is a misleading model-comparison tool when models are similarly accurate.** All three models scored within 1% of each other on RMSE, yet they make qualitatively different decisions about *which* features matter. Synthetic data makes that visible.
- **Lasso's "best test fit" was a coincidence of this particular noise seed.** Run with a different seed and OLS often wins by a hair. The robust signal is the coefficient-recovery story, not the third-decimal RMSE difference.
- **Standardizing features before Ridge / Lasso is not optional.** Without `StandardScaler`, the L1 / L2 penalty implicitly weights features by their natural scale, which is almost never what you want.
- **The Lasso regularization path is the single most informative diagnostic for feature selection.** It shows you exactly the α range where each feature transitions from "kept" to "dropped" — much more useful than picking one α and reporting one set of coefficients.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
