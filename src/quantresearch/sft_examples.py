"""Export teacher-selected tool-action examples; does NOT train any LLM.

The teacher exhaustively checks training-task *development* data to create
noisy supervision for the first hypothesis-selection step. No evaluation
seeds or test-slice performance are used to construct these examples.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

if __package__:
    from .market import generate_task, observed_features
    from .backtest import CANDIDATES, candidate_development_results, utility
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from quantresearch.market import generate_task, observed_features
    from quantresearch.backtest import CANDIDATES, candidate_development_results, utility


def build_examples(n: int, output: Path, start_seed: int = 500) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for j in range(n):
            task = generate_task(start_seed + j)
            scores = [utility(result) for result in candidate_development_results(task)]
            best_idx = int(np.argmax(scores))
            best_action = CANDIDATES[best_idx].name if scores[best_idx] > 0 else "STOP"
            obs = observed_features(task)
            row = {
                "kind": "candidate-supervised-tool-selection-example",
                "source": "synthetic-training-task-development-only",
                "messages": [
                    {"role": "system", "content": "Choose the next cost-aware quantitative research backtest from whitelisted actions. Reply with an action name only."},
                    {"role": "user", "content": json.dumps({
                        "train_autocorrelation": round(float(obs[1]/4), 4),
                        "train_daily_volatility_scaled": round(float(obs[2]), 4),
                        "cost_bps": task.cost_bps,
                        "allowed_actions": [c.name for c in CANDIDATES] + ["STOP"],
                    })},
                    {"role": "assistant", "content": best_action},
                ],
                "teacher_development_utility": round(float(max(0, scores[best_idx])), 4),
            }
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("artifacts/sft_candidates.jsonl"))
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be >=1")
    build_examples(args.examples, args.out)
    print(f"Wrote {args.examples} teacher-selected tool-action examples: {args.out}")


if __name__ == "__main__":
    main()
