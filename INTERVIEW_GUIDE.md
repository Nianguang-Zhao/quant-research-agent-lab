# Interview guide — Quant Research Agent Lab

## 60-second introduction (you can adapt this)

> I built a small RL-driven quantitative research harness to investigate a question relevant to Prime Intellect: can an agent learn how to conduct better experiments, not just maximize a backtest metric? The agent has a budget of three backtest calls and chooses among momentum and mean-reversion hypotheses under changing synthetic market regimes and trading costs. I implemented the environment, a linear softmax policy trained using REINFORCE, realistic-enough transaction-cost accounting, timestamp-provenance checks that reject look-ahead bias, and a chronological held-out evaluation. It also saves auditable traces and supports an optional local LLM tool-calling interface. The out-of-sample improvement over a strong simple heuristic was small and statistically inconclusive. I'd present that uncertainty as a finding, not hide it. My next step would be to replace the simple selector with a language-model agent and train on verified, long-horizon research trajectories.

## Live walkthrough (about 4 minutes)

**1. Train and evaluate**

```bash
python -m quantresearch.demo --train-episodes 900 --eval-tasks 80
```

Explain: the policy only reads training features and results of chosen development backtests; it never reads test performance while choosing actions. A newly seeded set of 80 market tasks is evaluated once after policy training.

**2. Show look-ahead defenses**

Open `backtest.py` and show `latest_source_bar[t] < t`. Point to the printed `PASS: verifier rejected oracle` message and `test_future_data_cannot_change_past_actions()`.

**3. Show the trace**

Open `artifacts/research_trace.jsonl`. It records each tool call, observed development score, reward, and updated hypothesis choice. Compare RL, heuristic and random on the same task.

**4. Show the metrics**

Open `artifacts/heldout_comparison.png` and `report.json`. RL has slightly higher mean Sharpe on the seeded synthetic tasks, but the confidence interval for the *paired difference* includes zero. Report the regime-shift and doubled-cost rows too.

## Likely questions & strong responses

**Q: Why RL instead of supervised learning?**  
Supervised learning could imitate teacher strategies from labeled examples, and I also export candidate demonstrations. RL is useful when experimental sequences and the value of intermediate tool calls matter. A multi-step agent can revise its choice after seeing tool results. This toy setting is small enough that a heuristic often competes, so I'd only justify RL after showing a controlled benefit.

**Q: What is the Markov decision process?**  
A research episode is one task. The observation combines train-only autocorrelation, volatility, trading cost, elapsed research budget and current best development score. An action is a hypothesis-tool call or STOP. The transition adds the verified backtest result to the research history and changes the allowed-action mask. Reward is improvement in best bounded development utility less computational research cost. The episode terminates after three calls or STOP.

**Q: Did you fine-tune an LLM?**  
No. The RL component is a trained linear softmax research policy, and there is a separate optional local LLM agent. The SFT file generator creates training examples but does not fine-tune language-model weights. A next version could perform SFT followed by PPO/GRPO-style post-training on verified LLM tool-use trajectories. I'd say this clearly rather than mischaracterize the project.

**Q: How do you prevent reward hacking?**  
A timestamp verifier rejects same-bar leakage. Rewards use net-of-cost Sharpe with a drawdown penalty and clipping, and the agent's calls are restricted to a whitelist with finite budget. The final lockbox is separate. These defenses are incomplete: the fixed simulator can still be exploited, and model improvements must be checked under perturbations and independent ground-truth evaluations.

**Q: Why synthetic data?**  
Synthetic, regime-controlled tasks make reproducible debug experiments possible, including intentional regime shifts and known source of failure. I explicitly avoid interpreting synthetic performance as market alpha. The CSV loader lets me probe transfer to real price data, but a serious real-market study needs point-in-time corporate-action and exchange data plus realistic fills.

**Q: Did RL outperform the heuristic?**  
Not conclusively. In the seeded example, average test Sharpe was approximately 0.403 vs 0.354, but the paired confidence interval included zero. That suggests more experiments, alternative reward structures, larger action spaces, or more expressive policies are needed. It may also tell us the problem is adequately solved by a simpler baseline.

**Q: What would you do to make this Prime Intellect-grade?**  
I would make the environment a secure tool sandbox with strict point-in-time data permissions, replace the simplified backtest with a genuine event-driven fill simulator, use an open-weight LLM for multi-step research and code generation, collect verified trajectories, compare SFT and RL post-training against identical inference budgets, and instrument failures across model policy, reward verifier, data, and infrastructure.

## What not to claim

- Do **not** claim the model learned profitable live trading strategies.
- Do **not** claim the LLM was RL-fine-tuned or deployed; it was not.
- Do **not** imply test performance drove policy training.
- Do **not** call the fixed turnover-cost assumption a realistic exchange matching engine.
- Do **not** say the tiny Sharpe difference is statistically proven.

## Credible resume bullet (once you've run and understood it)

> Developed a Python RL research-agent sandbox using policy-gradient training, tool-based strategy backtesting, signal-provenance verification, transaction-cost modeling, and untouched out-of-sample evaluation; benchmarked against heuristic and random agents across synthetic market regimes.
