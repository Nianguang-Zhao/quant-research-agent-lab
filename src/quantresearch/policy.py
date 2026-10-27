"""Actual policy-gradient post-training of the research-tool selector.

This trains a seven-action linear softmax policy with REINFORCE; it does NOT
fine-tune LLM weights. The interface allows a future LLM tool selector to
replace the linear policy while reusing envs, rewards and verifiers.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .env import ResearchEnv, N_ACTIONS, STOP
from .market import MarketTask

N_FEATURES = 7


class Policy:
    def __init__(self, seed: int = 17):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.025, (N_ACTIONS, N_FEATURES))

    def probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = self.weights @ obs
        logits = logits - np.max(logits[mask])
        exp = np.exp(np.clip(logits, -100, 40)) * mask
        return exp / np.sum(exp)

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
            greedy: bool = False) -> tuple[int, np.ndarray]:
        p = self.probs(obs, mask)
        action = int(np.argmax(p)) if greedy else int(rng.choice(N_ACTIONS, p=p))
        return action, p

    def save(self, path: str) -> None:
        np.savez_compressed(path, weights=self.weights)

    @classmethod
    def load(cls, path: str) -> "Policy":
        agent = cls()
        with np.load(path) as data:
            w = data["weights"]
        if w.shape != (N_ACTIONS, N_FEATURES):
            raise ValueError("Unexpected policy weights shape")
        agent.weights = w
        return agent


@dataclass
class TrainMetrics:
    mean_reward_last_100: float
    mean_reward_first_100: float
    mean_actions_last_100: float
    episodes: int


def train_policy(tasks: list[MarketTask], *, lr: float = 0.025, seed: int = 43,
                 gamma: float = 0.98) -> tuple[Policy, TrainMetrics]:
    """REINFORCE with an exponential-moving-average baseline.

    Each episode gets a separately seeded synthetic task. The objective is
    cumulative *development* research reward, never lockbox results.
    """
    rng = np.random.default_rng(seed)
    policy = Policy(seed)
    baseline = 0.0
    episode_rewards: list[float] = []
    num_actions: list[int] = []
    for task in tasks:
        env = ResearchEnv(task)
        records = []
        done = False
        while not done:
            obs = env.observe()
            action, probs = policy.act(obs, env.mask(), rng, greedy=False)
            _, reward, done, _ = env.step(action)
            records.append((obs, action, probs, reward))
        G = 0.0
        gradient = np.zeros_like(policy.weights)
        for obs, action, probs, reward in reversed(records):
            G = reward + gamma * G
            advantage = np.clip(G - baseline, -5, 5)
            onehot = np.zeros(N_ACTIONS)
            onehot[action] = 1
            gradient += advantage * np.outer(onehot - probs, obs)
        # Simple gradient norm clipping.
        norm = np.linalg.norm(gradient)
        if norm > 10:
            gradient *= 10 / norm
        policy.weights += lr * gradient
        episode_r = sum(x[3] for x in records)
        baseline = 0.985 * baseline + 0.015 * episode_r
        episode_rewards.append(episode_r)
        num_actions.append(env.steps)
    n = min(100, len(episode_rewards))
    return policy, TrainMetrics(float(np.mean(episode_rewards[-n:])),
                                float(np.mean(episode_rewards[:n])),
                                float(np.mean(num_actions[-n:])), len(tasks))


def run_episode(task: MarketTask, policy: Policy | None, *,
                mode: str = "rl", seed: int = 123) -> ResearchEnv:
    """Run a frozen research agent; 'random' and 'heuristic' are baselines."""
    rng = np.random.default_rng(seed)
    env = ResearchEnv(task)
    while not env.done:
        if mode == "rl":
            if policy is None:
                raise ValueError("RL mode requires policy")
            action, _ = policy.act(env.observe(), env.mask(), rng, greedy=True)
        elif mode == "random":
            # Same experiment budget as RL, random candidate order; never stop early.
            choices = np.flatnonzero(env.mask()[:STOP])
            action = int(rng.choice(choices))
        elif mode == "heuristic":
            # Intuitive domain baseline: prioritize the sign of estimated autocorr,
            # trying 3 thresholds in sequence, with no insight from test data.
            direction_trend = env.observe()[1] >= 0
            options = [0, 1, 2] if direction_trend else [3, 4, 5]
            action = next((i for i in options if i not in env.tried), STOP)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        env.step(action)
    return env
