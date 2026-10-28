# Quant Research Agent Lab


**Scientific honesty:** This repository trains a small **linear policy-gradient RL agent to choose quantitative research tools**. It includes an **optional local LLM tool-calling agent** and exports **SFT-ready *examples***. It does **not** claim to have fine-tuned an LLM, found real trading alpha, or deployed live orders. The default market data are synthetic. The distinction is important in an interview.

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


