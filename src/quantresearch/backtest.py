"""Signal provenance and transaction-cost-aware, chronological evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .market import MarketTask


@dataclass(frozen=True)
class Candidate:
    name: str
    direction: int  # +1 = trend following, -1 = mean reversion
    threshold: float  # signal only if |prior return| > threshold * past 20d vol
    lag: int = 1  # earliest legal decision source is the previous bar


CANDIDATES = tuple(
    Candidate(f"{name}_z{threshold:g}", direction, threshold)
    for name, direction in (("trend", 1), ("revert", -1))
    for threshold in (0.0, 0.7, 1.4)
)


class LookaheadError(ValueError):
    pass


@dataclass
class Signal:
    position: np.ndarray
    latest_source_bar: np.ndarray


def make_signal(returns: np.ndarray, candidate: Candidate) -> Signal:
    n = len(returns)
    src = np.arange(n) - candidate.lag
    p = np.zeros(n, dtype=float)
    # Explicit provenance: which latest return was available when trading bar t?
    valid = (src >= 0) & (src < n)
    prev_r = np.zeros(n, dtype=float)
    prev_r[valid] = returns[src[valid]]
    # This trailing std is explicitly shifted by one full bar.
    trailing_vol = pd.Series(returns).shift(1).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    active = valid & np.isfinite(trailing_vol) & (np.abs(prev_r) > candidate.threshold * trailing_vol)
    p[active] = candidate.direction * np.sign(prev_r[active])
    # Position depends on BOTH the signal return and the lagged volatility.
    # Record the later of their source bars, not just the price signal's.
    latest_source = np.maximum(src, np.arange(n) - 1)
    return Signal(p, latest_source)


def verify_signal(signal: Signal) -> None:
    """Hard gate: reject any nonzero order whose data source is not in the past."""
    if not np.isfinite(signal.position).all():
        raise ValueError("Nonfinite positions")
    if np.any(np.abs(signal.position) > 1):
        raise ValueError("Position magnitude above permitted leverage")
    idx = np.arange(len(signal.position))
    traded = signal.position != 0
    if np.any(signal.latest_source_bar[traded] >= idx[traded]):
        raise LookaheadError("LOOK-AHEAD: a position depended on same-bar or future data")
    if np.any(signal.latest_source_bar[traded] < 0):
        raise LookaheadError("Invalid source timestamps")


@dataclass
class BacktestResult:
    sharpe: float
    total_return: float
    max_drawdown: float
    mean_turnover: float
    net_returns: np.ndarray
    equity: np.ndarray
    n_bars: int


def run_backtest(task: MarketTask, candidate: Candidate, start: int, end: int) -> BacktestResult:
    """Signals use previous completed bars; costs charged at position changes.

    Simplification: close-to-close exposure and fixed per-unit turnover cost.
    This is NOT an exchange-grade fill, spread, impact or capacity simulator.
    """
    if start < 0 or end > task.n or end <= start:
        raise ValueError("Invalid backtest window")
    sig = make_signal(task.returns, candidate)
    verify_signal(sig)
    position = sig.position[start:end]
    lagged_position = np.r_[0.0, position[:-1]]
    turnover = np.abs(position - lagged_position)
    cost_per_unit = task.cost_bps / 10000
    pnl = position * task.returns[start:end] - turnover * cost_per_unit
    pnl = pnl.copy()
    pnl[-1] -= abs(position[-1]) * cost_per_unit  # liquidation at window end
    equity = np.cumprod(1 + pnl)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    dd = float(np.max(1 - equity / peak))
    std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = float(np.sqrt(252) * np.mean(pnl) / std) if std > 1e-12 else 0.0
    return BacktestResult(sharpe, float(equity[-1] - 1), dd,
                          float(np.mean(turnover)), pnl, equity, len(pnl))


def utility(result: BacktestResult) -> float:
    """Observable evaluation proxy used for research reward (not live P&L)."""
    # Bounded to prevent one spectacular backtest from dominating RL updates.
    return float(np.clip(result.sharpe, -4, 4) - 0.5 * result.max_drawdown / 0.15)


def candidate_development_results(task: MarketTask) -> list[BacktestResult]:
    return [run_backtest(task, c, task.train_end, task.valid_end) for c in CANDIDATES]
