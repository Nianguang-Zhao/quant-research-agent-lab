import numpy as np
import pytest
from quantresearch.market import generate_task, MarketTask, observed_features
from quantresearch.backtest import (
    CANDIDATES, Candidate, LookaheadError, make_signal, verify_signal,
    run_backtest,
)


def test_same_bar_oracle_rejected():
    task = generate_task(101)
    oracle = Candidate("oracle", +1, 0.0, lag=0)
    with pytest.raises(LookaheadError):
        verify_signal(make_signal(task.returns, oracle))


def test_legitimate_lagged_signal_passes():
    task = generate_task(101)
    verify_signal(make_signal(task.returns, CANDIDATES[0]))


def test_future_data_cannot_change_past_actions():
    task = generate_task(42)
    first = make_signal(task.returns, CANDIDATES[2])
    revised = task.returns.copy()
    revised[350:] = 100  # nonsense future values may NOT alter earlier signals
    second = make_signal(revised, CANDIDATES[2])
    np.testing.assert_array_equal(first.position[:350], second.position[:350])


def test_no_test_data_used_in_features():
    original = generate_task(12)
    modified = original.returns.copy()
    modified[original.valid_end:] *= -200
    other = MarketTask(modified, original.train_end, original.valid_end,
                       original.cost_bps, original.source)
    np.testing.assert_array_equal(observed_features(original), observed_features(other))


def test_costs_matter():
    task = generate_task(222)
    cheap = MarketTask(task.returns, task.train_end, task.valid_end, 0, "test")
    costly = MarketTask(task.returns, task.train_end, task.valid_end, 20, "test")
    a = run_backtest(cheap, CANDIDATES[0], cheap.train_end, cheap.valid_end)
    b = run_backtest(costly, CANDIDATES[0], costly.train_end, costly.valid_end)
    assert a.total_return > b.total_return
    assert a.mean_turnover > 0
