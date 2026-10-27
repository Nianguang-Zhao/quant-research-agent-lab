"""A small sequential research-agent environment. Not a live trading environment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .market import MarketTask, observed_features
from .backtest import CANDIDATES, BacktestResult, candidate_development_results, utility

STOP = len(CANDIDATES)
N_ACTIONS = STOP + 1
BUDGET = 3
COMPUTE_PENALTY = 0.06  # each evaluated candidate consumes research budget


@dataclass
class Step:
    action: int
    candidate: str
    dev_sharpe: float | None
    dev_utility: float | None
    reward: float
    best_so_far: str | None


class ResearchEnv:
    """Choose up to three hypothesis backtests, or stop early.

    Agents can only see data from the train slice and results from development
    experiments they explicitly called. Test data is not accessed here.
    """

    def __init__(self, task: MarketTask, budget: int = BUDGET):
        self.task = task
        self.budget = budget
        self._dev: list[BacktestResult] = candidate_development_results(task)
        self._features = observed_features(task)
        self.reset()

    def reset(self) -> np.ndarray:
        self.tried: set[int] = set()
        self.best_idx: int | None = None
        self.best_score = 0.0  # flat benchmark, utility zero
        self.steps = 0
        self.done = False
        self.trace: list[Step] = []
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.r_[self._features, self.steps / self.budget, self.best_score / 4]

    def mask(self) -> np.ndarray:
        m = np.ones(N_ACTIONS, dtype=bool)
        for i in self.tried:
            m[i] = False
        if self.steps >= self.budget:
            m[:] = False
        m[STOP] = True
        return m

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already finished")
        if action < 0 or action >= N_ACTIONS or not self.mask()[action]:
            raise ValueError("Invalid/repeated research action")
        if action == STOP:
            self.done = True
            self.trace.append(Step(action, "STOP", None, None, 0.0,
                                   CANDIDATES[self.best_idx].name if self.best_idx is not None else None))
            return self.observe(), 0.0, True, {"action": "STOP"}
        result = self._dev[action]
        score = utility(result)
        improvement = max(0, score - self.best_score)
        reward = improvement - COMPUTE_PENALTY
        if score > self.best_score:
            self.best_score = score
            self.best_idx = action
        self.tried.add(action)
        self.steps += 1
        self.done = self.steps >= self.budget
        selected = CANDIDATES[self.best_idx].name if self.best_idx is not None else None
        self.trace.append(Step(action, CANDIDATES[action].name,
                               float(result.sharpe), float(score), float(reward), selected))
        return self.observe(), float(reward), self.done, {"tested": CANDIDATES[action].name}

    def selected(self) -> str:
        return CANDIDATES[self.best_idx].name if self.best_idx is not None else "FLAT"

    def lockbox_result(self) -> BacktestResult | None:
        """Call ONLY AFTER research selection has finished; never for training."""
        if not self.done:
            raise RuntimeError("Lockbox cannot be opened during research")
        if self.best_idx is None:
            return None
        from .backtest import run_backtest
        return run_backtest(self.task, CANDIDATES[self.best_idx], self.task.valid_end, self.task.n)
