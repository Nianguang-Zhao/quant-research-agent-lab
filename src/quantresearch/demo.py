"""One-command train/evaluate/report demo, with fully held-out final testing."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np  # type: ignore[import-not-found]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .market import generate_task, load_close_csv
from .policy import train_policy, run_episode
from .backtest import CANDIDATES, Candidate, LookaheadError, make_signal, verify_signal, run_backtest


def evaluate(tasks, policy) -> tuple[dict, dict]:
    modes = ["rl", "heuristic", "random"]
    collected = {name: [] for name in modes}
    demo_traces = {}
    for j, task in enumerate(tasks):
        for mode in modes:
            env = run_episode(task, policy, mode=mode, seed=20_000+j)
            bt = env.lockbox_result()
            result = {
                "task": j, "strategy": env.selected(), "experiments": env.steps,
                "lockbox_sharpe": bt.sharpe if bt is not None else 0.0,
                "lockbox_return": bt.total_return if bt is not None else 0.0,
                "lockbox_drawdown": bt.max_drawdown if bt is not None else 0.0,
                "regime_shift": bool(task.true_phi_train * task.true_phi_test < 0)
                                 if task.true_phi_train is not None else None,
                "stress_double_cost_sharpe": (
                    run_backtest(
                        type(task)(task.returns, task.train_end, task.valid_end,
                                   task.cost_bps * 2, task.source,
                                   task.true_phi_train, task.true_phi_test),
                        CANDIDATES[env.best_idx],
                        task.valid_end, task.n,
                    ).sharpe if env.best_idx is not None else 0.0
                ),
            }
            collected[mode].append(result)
            if j == 0:
                demo_traces[mode] = {
                    "market_source": task.source,
                    "chronological_split": task.split,
                    "prior_train_autocorrelation": float(np.corrcoef(task.returns[1:task.train_end], task.returns[:task.train_end-1])[0,1]),
                    "one_way_cost_bps_per_position_unit": task.cost_bps,
                    "steps": [asdict(s) for s in env.trace],
                    "selected_strategy": env.selected(),
                    "lockbox_result": result,
                }
    summary = {}
    bootstrap_rng = np.random.default_rng(2026)
    for mode, rows in collected.items():
        s = np.array([row["lockbox_sharpe"] for row in rows])
        r = np.array([row["lockbox_return"] for row in rows])
        resamples = bootstrap_rng.integers(0, len(s), size=(2000, len(s)))
        low, high = np.quantile(s[resamples].mean(axis=1), [0.025, 0.975])
        shifted = [row["lockbox_sharpe"] for row in rows if row["regime_shift"]]
        stable = [row["lockbox_sharpe"] for row in rows if row["regime_shift"] is False]
        summary[mode] = {
            "mean_sharpe_bootstrap_95ci": [round(float(low), 3), round(float(high), 3)],
            "mean_sharpe_regime_shift": round(float(np.mean(shifted)), 3) if shifted else None,
            "mean_sharpe_no_shift": round(float(np.mean(stable)), 3) if stable else None,
            "mean_sharpe_double_trading_cost": round(float(np.mean([row["stress_double_cost_sharpe"] for row in rows])), 3),
            "mean_lockbox_sharpe": round(float(s.mean()), 3),
            "median_lockbox_sharpe": round(float(np.median(s)), 3),
            "mean_lockbox_return": round(float(r.mean()), 4),
            "fraction_positive_lockbox_return": round(float((r > 0).mean()), 3),
            "mean_experiments": round(float(np.mean([row["experiments"] for row in rows])), 2),
        }
    # Paired comparison on identical held-out tasks, not independent samples.
    differences = np.array([a["lockbox_sharpe"] - b["lockbox_sharpe"]
                            for a, b in zip(collected["rl"], collected["heuristic"])])
    resamples = bootstrap_rng.integers(0, len(differences), size=(2000, len(differences)))
    lo, hi = np.quantile(differences[resamples].mean(axis=1), [0.025, 0.975])
    paired = {
        "rl_minus_heuristic_mean_sharpe": round(float(differences.mean()), 3),
        "paired_bootstrap_95ci": [round(float(lo), 3), round(float(hi), 3)],
        "caution": "If interval includes zero, do not claim a statistically established RL advantage.",
    }
    return {"summary": summary, "paired_rl_vs_heuristic": paired,
            "per_task": collected}, demo_traces


def leak_check() -> str:
    task = generate_task(77)
    # The illicit agent tries to use the current bar's return to trade that bar.
    fake = Candidate("impossible_same_bar_oracle", +1, 0.0, lag=0)
    try:
        verify_signal(make_signal(task.returns, fake))
    except LookaheadError as exc:
        return f"PASS: verifier rejected oracle: {exc}"
    raise AssertionError("CRITICAL: look-ahead verifier failed")


def save_chart(report, filename: Path) -> None:
    modes = ["random", "heuristic", "rl"]
    vals = [report["summary"][mode]["mean_lockbox_sharpe"] for mode in modes]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ci = [report["summary"][mode]["mean_sharpe_bootstrap_95ci"] for mode in modes]
    error_bars = np.array([[max(0, y-low) for y, (low, high) in zip(vals, ci)],
                           [max(0, high-y) for y, (low, high) in zip(vals, ci)]])
    rects = ax.bar(modes, vals, yerr=error_bars, capsize=5,
                   color=["#a7adb7", "#5e91ac", "#2563eb"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean out-of-sample Sharpe (synthetic tasks)")
    ax.set_title("Frozen policies on unseen task seeds / untouched lockbox")
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    for rect, value in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{value:+.2f}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    fig.text(.5, .015, "Illustrative synthetic experiment; not investable performance or evidence of LLM fine-tuning.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-episodes", type=int, default=900)
    p.add_argument("--eval-tasks", type=int, default=80)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--csv", type=str, default=None,
                   help="Optional separate analysis for Date,Close daily market CSV. Not mixed with RL training.")
    args = p.parse_args(argv)
    if args.train_episodes < 10 or args.eval_tasks < 5:
        p.error("Use >=10 training episodes and >=5 evaluation tasks")
    args.out.mkdir(parents=True, exist_ok=True)
    train = [generate_task(i + 500) for i in range(args.train_episodes)]
    policy, metrics = train_policy(train)
    # Completely disjoint seeds: these tasks never contribute to policy gradients.
    heldout = [generate_task(i + 50_000) for i in range(args.eval_tasks)]
    evaluated, traces = evaluate(heldout, policy)
    integrity_result = leak_check()
    report = {
        "experiment": "RL policy-gradient research-tool selection (NOT LLM post-training)",
        "dataset": "SYNTHETIC AR(1) returns, occasionally shifting in lockbox test",
        "train_episodes": args.train_episodes,
        "evaluation_tasks": args.eval_tasks,
        "train_reward_first_100": round(metrics.mean_reward_first_100, 3),
        "train_reward_last_100": round(metrics.mean_reward_last_100, 3),
        "train_actions_last_100": round(metrics.mean_actions_last_100, 3),
        "verification": integrity_result,
        "evaluation": evaluated,
        "limitations": [
            "Artificial daily close-to-close returns with engineered autocorrelation",
            "Fixed turnover cost, not order-book/latency/market-impact simulation",
            "No live trading, no real-world alpha claims, no LLM fine-tuning",
            "RL policy is a linear softmax network; results vary by seed",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    with (args.out / "research_trace.jsonl").open("w") as f:
        for name, trajectory in traces.items():
            f.write(json.dumps({"agent": name, **trajectory}) + "\n")
    policy.save(str(args.out / "policy.npz"))
    save_chart(evaluated, args.out / "heldout_comparison.png")
    print("\nQuant Research Agent Lab — reproducible results")
    print("--------------------------------------------")
    print(f"Training episodes: {args.train_episodes}; NEW evaluation tasks: {args.eval_tasks}")
    print(f"Mean training episode reward, early -> late: {metrics.mean_reward_first_100:.3f} -> {metrics.mean_reward_last_100:.3f}")
    for mode, row in evaluated["summary"].items():
        print(f"{mode:10} | holdout Sharpe {row['mean_lockbox_sharpe']:+.3f} | "
              f"positive holdout {row['fraction_positive_lockbox_return']:.1%} | "
              f"research experiments {row['mean_experiments']:.2f}")
    pair = evaluated["paired_rl_vs_heuristic"]
    print(f"Paired RL−heuristic lockbox Sharpe: {pair['rl_minus_heuristic_mean_sharpe']:+.3f} "
          f"(bootstrap 95% interval {pair['paired_bootstrap_95ci']})")
    print(integrity_result)
    print(f"Files: {args.out.resolve()}")
    if args.csv:
        real_task = load_close_csv(args.csv)
        actual = {mode: run_episode(real_task, policy, mode=mode, seed=99) for mode in ("rl", "heuristic", "random")}
        print("\nUSER-CSV EXPERIMENT — exploratory ONLY; no live trading claim")
        for mode, env in actual.items():
            bt = env.lockbox_result()
            print(f"{mode:10} | selected {env.selected():14} | "
                  f"lockbox Sharpe {bt.sharpe if bt else 0.0:+.3f}")


if __name__ == "__main__":
    main()
