# Quant Research Agent Lab

**An executable interview project at the intersection of reinforcement learning, tool-using agents, quantitative research, and post-training evaluation.**

> **Scientific honesty:** This repository trains a small **linear policy-gradient RL agent to choose quantitative research tools**. It includes an **optional local LLM tool-calling agent** and exports **SFT-ready *examples***. It does **not** claim to have fine-tuned an LLM, found real trading alpha, or deployed live orders. The default market data are synthetic. The distinction is important in an interview.

## What it demonstrates

A quantitative research agent has a budget of **3 backtest calls**. It must choose from six plausible strategies (trend or reversal, each with three thresholds), assess the evidence from development-period backtests, or stop early. After the selection is frozen, a **chronological lockbox** evaluates the selected strategy out of sample.

The agent is not rewarded for a visually impressive in-sample equity curve. It receives a *bounded*, cost-aware, risk-penalized development score, less a penalty for each backtest call. A separate verifier rejects look-ahead signals regardless of profitability.

- **Research environment:** seeded, regime-varying synthetic returns; optionally user-supplied daily-close CSV.
- **Tools:** six hardcoded candidate backtests, with position constraints and fixed transaction costs. An optional LLM chooses tools using local OpenAI-compatible chat completions; generated arbitrary Python is **not executed**.
- **RL:** actual REINFORCE policy gradient with EMA baseline to learn which hypothesis to test next; finite tool budgets and masks for already-used actions.
- **Evaluations:** chronological train / development / lockbox, two rule-based baselines, stress at twice the trading cost, regime-shift diagnostics, paired bootstrap confidence intervals.
- **Post-training interface:** export teacher-selected training examples in chat-message JSONL, suitable as a starting point for subsequent **separate** LLM SFT experiments.
- **Auditing:** per-step JSONL traces, data-provenance checks, hard rejection of same-bar oracle signals, and unit tests.

## Architecture

```text
Synthetic market generator or user-supplied CSV
                |
   Chronological data partitions
      TRAIN | DEVELOPMENT | TEST (LOCKBOX)
        |          |          ^
        v          v          | (only after agent finishes)
   State summary   Backtest tool + verifier
        |             ^
        v             |
  RL Research Agent (3-tool-call budget)
        |             |
        +---- selects one hypothesis
        |
        +---- reads tool results, updates research state
        |
        +---- STOP / budget exhausted --> freeze selected hypothesis
                                      |
                                LOCKBOX EVALUATION
                                      |
                 compare RL / heuristic / random / stress
```

**Key separation:** TRAIN features determine the agent's starting state; DEVELOPMENT data can be used by the tool evaluator to select a hypothesis; the TEST slice is never used to choose tools or compute gradients. The frozen policy is trained on synthetic tasks with seeds 500+ and evaluated on disjoint tasks with seeds 50,000+.

## Run in a few commands

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

pytest -q
python -m quantresearch.demo --train-episodes 900 --eval-tasks 80
```

This creates:

```text
artifacts/
  report.json                 full metrics, paired confidence interval, test details
  research_trace.jsonl        replayable three-agent reasoning/tool trajectories
  heldout_comparison.png      visualization with bootstrap error bars
  policy.npz                  trained policy weights
```

To produce candidate supervised post-training examples from **training-only** synthetic tasks:

```bash
python -m quantresearch.sft_examples --examples 200
```

They are **candidate supervisory examples, not evidence of completed LLM SFT**.

### Expected results for the bundled seeded experiment

The current example run (900 training episodes; 80 new evaluation tasks) produced:

| Evaluation metric | RL research policy | Autocorrelation heuristic | Random candidate testing |
|---|---:|---:|---:|
| Mean untouched-lockbox Sharpe | **0.403** | 0.354 | 0.153 |
| Positive lockbox return (task fraction) | 52.5% | 52.5% | 47.5% |
| Average candidate tool calls | 3.0 | 3.0 | 3.0 |

The paired RL-minus-heuristic Sharpe difference was **+0.048**, but its bootstrap 95% interval was **[−0.103, +0.213]**. **It is not statistically established that RL outperforms the simple heuristic.** This is an excellent prompt for a real research discussion: maybe more training helps, or maybe the added RL complexity isn't justified for this toy problem. You should not advertise these synthetic results as an investable strategy.

For a detailed metrics file, inspect `artifacts/report.json` after running.

## Optional: Local LLM agent through Ollama

If you already have Ollama installed and want to compare an LLM tool-calling agent to the trained RL selector, you can run (requires Ollama and an available local model):

```bash
ollama pull qwen3:1.7b
ollama serve                         # only if your Ollama server is not running
# in another terminal:
python -m quantresearch.llm_agent --model qwen3:1.7b
```

Default local endpoint: `http://localhost:11434/v1/chat/completions`. Custom endpoint:

```bash
LLM_API_KEY='your-key' python -m quantresearch.llm_agent \
  --url https://your-openai-compatible-host/v1/chat/completions \
  --model your-model-name
```

**This is an inference-time agent, NOT an RL-post-trained LLM.** The model receives a constrained set of valid tool names and development feedback. No model-generated Python is executed. The exact local API behavior depends on the installed model/version. Never send proprietary market data to an external endpoint without authorization.

## Optional: Your own daily prices

Create a CSV with headers `Date,Close` and **at least 211 trading-day closes**. Prefer total-return-adjusted closes to account for corporate actions:

```csv
Date,Close
2023-01-03,100.02
2023-01-04,100.50
2023-01-05,99.90
```

Then run:

```bash
python -m quantresearch.demo --csv path/to/daily_adjusted_prices.csv
```

The RL policy was trained **only on synthetic market tasks**. The CSV experiment is an **out-of-distribution diagnostic**, not evidence that the policy works on SPY or real markets. The signal is based on the prior day's closing return, and the simplified model assumes close-to-close exposure without detailed closing auction or overnight fill modeling.

## Three safeguards to show in an interview

### 1. Data leakage / look-ahead verifier

For strategy positions at bar `t`, the provenance must satisfy:

```text
latest_source_bar[t] < t
```

If a candidate tries to trade bar `t` using information from `t` (the impossible “oracle”), the verifier raises `LookaheadError` before a backtest is accepted. `tests/test_backtest.py` also checks that editing **future** returns does not alter earlier signals.

### 2. Realistic enough execution to invalidate naive wins

Daily net return uses:

```text
strategy_return[t] = position[t] * asset_return[t]
                   - abs(position[t] - previous_position[t]) * cost_per_unit
```

A terminal liquidation cost is added. Fixed transaction costs are illustrative and do **not** include full order-book dynamics, queue priority, market impact, latency, borrow fees, funding, overnight gap slippage, or capacity. Production work would require much stronger market mechanics.

### 3. Proper research vs. final evaluation

The RL agent is scored on **development utility**, not lockbox returns. It can adapt its sequence of backtest calls as it observes development feedback. The **final** selected strategy is evaluated on unseen lockbox data only after research selection is complete. Both regime shifts and doubled-cost stress tests are reported after freezing the choice.

## Where actual RL happens

The policy maps the seven-dimensional observation to seven actions (six candidate tools plus STOP) using a softmax. It is trained with the classical REINFORCE gradient:

\[
\nabla_\theta J \approx \sum_t (G_t-b)\nabla_\theta \log\pi_\theta(a_t\mid s_t),
\]

where `G_t` is discounted development research reward and `b` is an exponential-moving-average baseline. Repeated candidate actions are masked, and a fixed call budget limits search. The actual code lives in `src/quantresearch/policy.py`.

**Reward:** positive improvement in best *development* utility minus 0.06 per backtest call. Utility includes clipped annualized Sharpe and a drawdown penalty. `FLAT` is a valid baseline with utility 0; if no positive candidate appears, the agent can choose not to trade.

RL chooses **research actions**, not buy/sell orders. A *trading-execution RL project* would require a different MDP and simulator.

## Honest limitations and high-value next steps

1. **Synthetic edge:** AR(1) market regimes have constructed autocorrelation. They are useful for testing the experimental system, not for establishing empirical alpha.
2. **Simplified market mechanics:** use next-open fills, historical intraday bid/ask/order book, corporate actions, funding, impact and capacity before making live-asset claims.
3. **Too-small learning problem:** only six predefined hypotheses; REINFORCE learns a tool selector, not natural-language long-horizon reasoning. A simple statistical heuristic can be competitive.
4. **LLM post-training not yet performed:** to advance, replace the selector with an open-weight language model that emits constrained tool calls, collect verified rollouts, run SFT and then RL post-training with held-out evaluation and KL/reference policy controls. Measure gain due to the model rather than changes to the harness.
5. **Selection bias remains:** development data are used repeatedly for candidate comparison. A strict final lockbox and further rolling out-of-time tests remain necessary. No repeated iteration should optimize to the same opened lockbox.
6. **Simulation cannot be a security boundary:** the high-level interface only exposes allowed observations, but a production agent's sandbox must isolate raw private data and test partitions as separate permissions/services.

## Repository map

```text
src/quantresearch/
  market.py          synthetic regime tasks, chronological split, CSV loader
  backtest.py        candidate strategies, provenance verification, cost model
  env.py             sequential research tool environment and finite budget
  policy.py          policy-gradient RL training and two baselines
  demo.py            frozen evaluation, paired bootstrap, metrics and charts
  llm_agent.py       optional whitelisted tool selection via local LLM
  sft_examples.py    export teacher-selected examples (no LLM training)
tests/
  test_backtest.py   future-information, provenance, cost tests
  test_env.py        lockbox access, action budgets, training/serialization tests
INTERVIEW_GUIDE.md    exact demo script, defensible claims, questions
artifacts/            seeded example outputs
```

The core skill on display is **experimental infrastructure and rigorous failure attribution**, not a claimed profitable signal.
