"""
Train OLS / Ridge / Lasso on the synthetic dataset, evaluate, and produce
the dashboard figures used in README.md.

Color palette (kept consistent across all figures):
    background  : white
    grid / axes : light gray
    primary     : muted blue   (#3B6EA8)
    accent      : muted red    (#C04040)
    neutral     : medium gray  (#7A7A7A)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from generate_data import DataConfig, generate

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": COLOR_GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@dataclass
class FitResult:
    name: str
    coef: np.ndarray
    intercept: float
    rmse: float
    mae: float
    r2: float
    y_pred_test: np.ndarray


def fit_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[FitResult]:
    models = {
        "OLS": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=10_000),
    }
    out: list[FitResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        out.append(FitResult(
            name=name,
            coef=model.coef_,
            intercept=float(model.intercept_),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            mae=float(mean_absolute_error(y_test, y_pred)),
            r2=float(r2_score(y_test, y_pred)),
            y_pred_test=y_pred,
        ))
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_metrics_bar(results: list[FitResult], out_path: Path) -> None:
    metric_names = ["RMSE", "MAE", "R²"]
    values = np.array([
        [r.rmse, r.mae, r.r2] for r in results
    ])  # shape (n_models, 3)
    model_names = [r.name for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    colors = [COLOR_BLUE, COLOR_GRAY, COLOR_RED]
    for ax, mname, col in zip(axes, metric_names, range(3)):
        bars = ax.bar(model_names, values[:, col], color=colors,
                      edgecolor=COLOR_LIGHT_GRAY, linewidth=0.8)
        ax.set_title(mname)
        ax.tick_params(axis="x", labelrotation=15)
        # Annotate each bar with the value.
        for bar, v in zip(bars, values[:, col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=COLOR_TEXT)
        # Lift the y-limit so labels don't clip.
        ymin = min(0.0, values[:, col].min())
        ymax = values[:, col].max()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.18 + 0.05)

    fig.suptitle("Test-set performance across models",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_coefficient_recovery(
    beta_true: np.ndarray,
    results: list[FitResult],
    out_path: Path,
) -> None:
    n_features = len(beta_true)
    feature_names = [f"x{i+1}" for i in range(n_features)]
    x_pos = np.arange(n_features)
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)

    # True coefficients shown as a baseline (gray bars).
    ax.bar(x_pos - 1.5 * width, beta_true, width=width,
           label="True β", color=COLOR_LIGHT_GRAY,
           edgecolor=COLOR_GRAY, linewidth=0.6)

    colors = [COLOR_BLUE, COLOR_GRAY, COLOR_RED]
    offsets = [-0.5 * width, 0.5 * width, 1.5 * width]
    for r, col, off in zip(results, colors, offsets):
        ax.bar(x_pos + off, r.coef, width=width, label=r.name,
               color=col, edgecolor="white", linewidth=0.4)

    ax.axhline(0, color=COLOR_GRAY, linewidth=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Coefficient recovery: true β vs. estimated β")
    ax.legend(loc="best", ncol=4)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_pred_vs_actual(
    y_test: np.ndarray,
    results: list[FitResult],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    colors = [COLOR_BLUE, COLOR_GRAY, COLOR_RED]

    lo = float(min(y_test.min(), min(r.y_pred_test.min() for r in results)))
    hi = float(max(y_test.max(), max(r.y_pred_test.max() for r in results)))
    pad = (hi - lo) * 0.05
    lims = (lo - pad, hi + pad)

    for ax, r, col in zip(axes, results, colors):
        ax.scatter(y_test, r.y_pred_test, s=18, alpha=0.5,
                   color=col, edgecolor="white", linewidth=0.3)
        ax.plot(lims, lims, color=COLOR_RED, linewidth=1.2,
                linestyle="--", label="ideal y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual y")
        ax.set_title(f"{r.name}\nR² = {r.r2:.3f}")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Predicted ŷ")
    fig.suptitle("Predicted vs. actual on test set",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_lasso_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_informative: int,
    out_path: Path,
) -> None:
    alphas, coefs, _ = lasso_path(X_train, y_train, n_alphas=80)
    # coefs shape: (n_features, n_alphas)

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)

    for i in range(coefs.shape[0]):
        is_informative = i < n_informative
        ax.plot(
            alphas, coefs[i, :],
            color=COLOR_BLUE if is_informative else COLOR_RED,
            linewidth=1.6 if is_informative else 1.0,
            alpha=0.9 if is_informative else 0.55,
            label=("informative" if i == 0
                   else ("redundant" if i == n_informative
                         else None)),
        )

    ax.set_xscale("log")
    ax.invert_xaxis()  # left = strong regularization
    ax.set_xlabel("Lasso α  (left = strong regularization → right = weak)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Lasso regularization path: redundant coefficients shrink first")
    ax.axhline(0, color=COLOR_GRAY, linewidth=0.7)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_residuals(
    y_test: np.ndarray,
    results: list[FitResult],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    colors = [COLOR_BLUE, COLOR_GRAY, COLOR_RED]
    for ax, r, col in zip(axes, results, colors):
        residuals = y_test - r.y_pred_test
        ax.scatter(r.y_pred_test, residuals, s=16, alpha=0.5,
                   color=col, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color=COLOR_RED, linewidth=1.0, linestyle="--")
        ax.set_xlabel("Predicted ŷ")
        ax.set_title(r.name)
    axes[0].set_ylabel("Residual (y − ŷ)")
    fig.suptitle("Residual diagnostics: should look like noise around zero",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    cfg = DataConfig()
    X_df, y_series, beta_true = generate(cfg)

    # Standardize features (Ridge / Lasso are scale-sensitive).
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)
    y = y_series.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=cfg.seed
    )

    results = fit_models(X_train, y_train, X_test, y_test)

    # Print to stdout for the user / journal.
    print(f"\nDataset: n={cfg.n_samples}, p={cfg.n_features} "
          f"({cfg.n_informative} informative + {cfg.n_redundant} redundant), "
          f"corr={cfg.correlation}, noise_std={cfg.noise_std}")
    print("\nTest metrics:")
    print(f"  {'model':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    for r in results:
        print(f"  {r.name:<18} {r.rmse:>8.4f} {r.mae:>8.4f} {r.r2:>8.4f}")

    # Coefficient recovery error vs. the true beta — the headline metric for
    # synthetic-data regression: how close did we get to the truth?
    print("\nCoefficient recovery (L2 distance from true β):")
    for r in results:
        err = float(np.linalg.norm(r.coef - beta_true))
        print(f"  {r.name:<18} ‖β̂ − β_true‖₂ = {err:.4f}")

    # Persist a metrics summary so README values stay in sync with last run.
    summary = {
        "config": cfg.__dict__,
        "metrics": [
            {
                "model": r.name,
                "rmse": r.rmse,
                "mae": r.mae,
                "r2": r.r2,
                "coef_l2_error": float(np.linalg.norm(r.coef - beta_true)),
            }
            for r in results
        ],
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures.
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    fig_metrics_bar(results, assets / "01_metrics.png")
    fig_coefficient_recovery(beta_true, results, assets / "02_coef_recovery.png")
    fig_pred_vs_actual(y_test, results, assets / "03_pred_vs_actual.png")
    fig_residuals(y_test, results, assets / "04_residuals.png")
    fig_lasso_path(X_train, y_train, cfg.n_informative, assets / "05_lasso_path.png")

    print(f"\nFigures saved to: {assets.resolve()}")
    print(f"Metrics saved to:  results/metrics.json")


if __name__ == "__main__":
    main()
