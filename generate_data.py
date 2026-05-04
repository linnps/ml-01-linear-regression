"""
Synthetic regression data generator.

The whole point of using synthetic data here is that the *true* coefficients
are known. After training, we can check whether OLS / Ridge / Lasso
actually recover them — a much sharper diagnostic than holdout RMSE.

Generative process:
    y = X @ beta_true + noise

with controllable:
    - n_samples
    - n_informative features (non-zero true coefficient)
    - n_redundant features (zero true coefficient — Lasso should drop these)
    - feature correlation (collinearity stresses OLS, helps Ridge shine)
    - noise standard deviation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    n_samples: int = 1000
    n_informative: int = 5
    n_redundant: int = 5
    correlation: float = 0.6
    noise_std: float = 1.0
    seed: int = 42

    @property
    def n_features(self) -> int:
        return self.n_informative + self.n_redundant


def generate(cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    # Build a covariance matrix where every pair has the requested correlation.
    cov = np.full((cfg.n_features, cfg.n_features), cfg.correlation)
    np.fill_diagonal(cov, 1.0)

    # Sample correlated features.
    X = rng.multivariate_normal(mean=np.zeros(cfg.n_features), cov=cov, size=cfg.n_samples)

    # True coefficients: first n_informative are nonzero, rest are zero.
    beta_true = np.zeros(cfg.n_features)
    beta_true[: cfg.n_informative] = rng.uniform(-3.0, 3.0, size=cfg.n_informative)

    # Linear target with Gaussian noise.
    noise = rng.normal(0.0, cfg.noise_std, size=cfg.n_samples)
    y = X @ beta_true + noise

    feature_names = [f"x{i+1}" for i in range(cfg.n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="y")

    return X_df, y_series, beta_true


def save(out_dir: Path, X: pd.DataFrame, y: pd.Series, beta_true: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "X.csv", index=False)
    y.to_csv(out_dir / "y.csv", index=False)
    np.save(out_dir / "beta_true.npy", beta_true)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic regression dataset.")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-informative", type=int, default=5)
    p.add_argument("--n-redundant", type=int, default=5)
    p.add_argument("--correlation", type=float, default=0.6)
    p.add_argument("--noise-std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(
        n_samples=args.n_samples,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        correlation=args.correlation,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    X, y, beta_true = generate(cfg)
    save(args.out_dir, X, y, beta_true)

    print(f"Generated {len(X)} samples, {cfg.n_features} features "
          f"({cfg.n_informative} informative, {cfg.n_redundant} redundant).")
    print(f"Saved to: {args.out_dir.resolve()}")
    print(f"True non-zero coefficients (first {cfg.n_informative}):")
    for i, b in enumerate(beta_true[: cfg.n_informative]):
        print(f"  beta[{i+1}] = {b:+.4f}")


if __name__ == "__main__":
    main()
