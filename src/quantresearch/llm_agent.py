"""Optional LOCAL LLM interface for tool-using research agents.

Uses an OpenAI-compatible chat endpoint, e.g. an Ollama localhost server.
No LLM post-training occurs here: this tests *inference-time tool use* only.
The model is allowed to choose from whitelisted strategies, never run code.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from urllib.request import Request, urlopen
import numpy as np
from .backtest import CANDIDATES
from .env import ResearchEnv, STOP
from .market import generate_task, load_close_csv


SYSTEM = """You are a skeptical quant researcher exploring a tiny pre-defined strategy set.
You must select exactly ONE action in each turn from the allowed list.
A strategy action runs a cost-aware development backtest; STOP ends research.
You may run at most 3 distinct candidate backtests per task.
Trust only historical training features and development feedback supplied below;
never infer future results or claim live alpha. Prefer convincing, cost-aware evidence.
Respond ONLY with JSON: {"action":"candidate_name_or_STOP", "reason":"brief"}.
"""


def query_compatible_api(messages: list[dict], *, url: str, model: str,
                         api_key: str | None = None) -> dict:
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": 0.0, "stream": False}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, payload, headers)
    with urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    # Accommodate models that wrap the JSON response in Markdown.
    if content.strip().startswith("{"):
        return json.loads(content)
    match = re.search(r"\{[^{}]*\}", content, flags=re.S)
    if not match:
        raise ValueError(f"LLM did not return a JSON action: {content[:200]}")
    return json.loads(match.group(0))


def run_llm_research(task, *, url: str, model: str, api_key=None):
    env = ResearchEnv(task)
    initial = env.observe()
    transcript = []
    while not env.done:
        choices = [CANDIDATES[i].name for i in range(STOP) if env.mask()[i]] + ["STOP"]
        observations = {
            "market_source": task.source,
            "train_lag_one_autocorrelation": round(float(initial[1] / 4), 4),
            "train_volatility_scaled": round(float(initial[2]), 4),
            "trading_cost_bps_per_unit": task.cost_bps,
            "remaining_research_calls": env.budget-env.steps,
            "previous_experiments": [x.__dict__ for x in env.trace],
            "actions_allowed": choices,
        }
        answer = query_compatible_api(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps(observations)}],
            url=url, model=model, api_key=api_key)
        name = answer.get("action")
        if name not in choices:
            raise ValueError(f"Invalid LLM tool choice: {name!r} (valid: {choices})")
        action = STOP if name == "STOP" else next(i for i, c in enumerate(CANDIDATES) if c.name == name)
        env.step(action)
        transcript.append({"selection": name, "reason": answer.get("reason", ""),
                           "tool_result": env.trace[-1].__dict__})
    return env, transcript


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    p.add_argument("--model", default="qwen3:1.7b")
    p.add_argument("--csv", default=None, help="Optional daily Date,Close CSV")
    args = p.parse_args()
    task = load_close_csv(args.csv) if args.csv else generate_task(50_000)
    env, transcript = run_llm_research(task, url=args.url, model=args.model,
                                      api_key=os.environ.get("LLM_API_KEY"))
    print(json.dumps({"model": args.model, "rollout": transcript,
                      "selected": env.selected(),
                      "lockbox_sharpe": (env.lockbox_result().sharpe
                                         if env.best_idx is not None else 0)}, indent=2))


if __name__ == "__main__":
    main()
