"""Regime-varying *synthetic* returns, never represented as real SPY observations.

The true autocorrelation regime is hidden from the research agent; it sees
only statistics computed on the training slice. Development and test data
are separate and respect chronological order.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MarketTask:
    returns: np.ndarray
    train_end: int
    valid_end: int
    cost_bps: float
    source: str
    true_phi_train: float | None = None  # diagnostics only; never shown to policy
    true_phi_test: float | None = None

    @property
    def n(self) -> int:
        return len(self.returns)

    @property
    def split(self) -> dict:
        return {
            "train": [0, self.train_end],
            "development": [self.train_end, self.valid_end],
            "lockbox_test": [self.valid_end, self.n],
        }


def generate_task(seed: int, *, n_train: int = 252,
                  n_valid: int = 126, n_test: int = 126,
                  shift_probability: float = 0.20) -> MarketTask:
    """AR(1) regimes with volatility/cost variation and an occasional *test-only* flip.

    A controlled experiment for RL pipeline debugging, not market evidence.
    The hidden generator's phi is used for diagnostics, not by agents.
    """
    rng = np.random.default_rng(seed)
    sign = int(rng.choice([-1, 1]))
    magnitude = float(rng.choice([0.12, 0.22, 0.34]))
    phi = sign * magnitude
    cost = float(rng.choice([2.0, 5.0, 10.0, 16.0]))
    sigma = float(rng.choice([0.007, 0.011, 0.016]))
    test_phi = -phi if rng.random() < shift_probability else phi * 0.75
    n = n_train + n_valid + n_test
    r = np.zeros(n, dtype=float)
    shock = rng.standard_normal(n)
    for t in range(1, n):
        current_phi = phi if t < n_train + n_valid else test_phi
        current_sigma = sigma if t < n_train + n_valid else sigma * 1.25
        r[t] = current_phi * r[t-1] + current_sigma * shock[t]
    return MarketTask(r, n_train, n_train+n_valid, cost, "synthetic_ar1_regime", phi, test_phi)


def load_close_csv(path: str, *, cost_bps: float = 5.0) -> MarketTask:
    """Optional user-supplied adjusted daily-close CSV (Date,Close).

    Assumes Close is total-return adjusted if corporate actions matter. Rows
    are sorted by timestamp; daily returns are observed at the close.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if not {"Date", "Close"}.issubset(df.columns):
        raise ValueError("CSV must contain Date and Close columns")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates in CSV")
    df = df.sort_values("Date")
    prices = pd.to_numeric(df["Close"], errors="raise")
    if (prices <= 0).any() or prices.isna().any():
        raise ValueError("Close must have positive, nonmissing prices")
    returns = prices.pct_change().dropna().to_numpy(dtype=float)
    if len(returns) < 210:
        raise ValueError("Need at least 211 daily closing prices")
    n = len(returns)
    train_end = int(n * .5)
    valid_end = int(n * .75)
    return MarketTask(returns, train_end, valid_end, cost_bps, f"user_csv:{path}")


def observed_features(task: MarketTask) -> np.ndarray:
    """State features use ONLY the earliest (train) observations.

    Features: bias, lag-one autocorrelation, volatility, cost, mean-reversion
    evidence magnitude. There is no test-period information in this vector.
    """
    r = task.returns[:task.train_end]
    corr = float(np.corrcoef(r[1:], r[:-1])[0, 1]) if np.std(r) > 1e-10 else 0.0
    corr = float(np.nan_to_num(corr))
    return np.array([
        1.0,
        float(np.clip(corr * 4, -2, 2)),
        float(np.clip(np.std(r) / 0.015, 0, 3)),
        task.cost_bps / 10.0,
        float(np.clip(abs(corr) * 4, 0, 2)),
    ], dtype=float)
