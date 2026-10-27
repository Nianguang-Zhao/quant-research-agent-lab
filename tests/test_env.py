import pytest
from quantresearch.market import generate_task
from quantresearch.env import ResearchEnv, STOP
from quantresearch.policy import Policy, train_policy, run_episode


def test_lockbox_is_not_available_while_researching():
    env = ResearchEnv(generate_task(77))
    with pytest.raises(RuntimeError):
        env.lockbox_result()
    env.step(STOP)
    assert env.lockbox_result() is None


def test_agent_cannot_repeat_experiment():
    env = ResearchEnv(generate_task(8))
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)


def test_agent_has_budget_limit():
    env = ResearchEnv(generate_task(9))
    for a in (0, 1, 2):
        env.step(a)
    assert env.done and env.steps == 3
    with pytest.raises(RuntimeError):
        env.step(3)
    assert env.lockbox_result() is not None or env.selected() == "FLAT"


def test_rl_training_and_policy_serialization(tmp_path):
    tasks = [generate_task(i) for i in range(12)]
    agent, metrics = train_policy(tasks)
    assert metrics.episodes == 12
    path = tmp_path / "policy.npz"
    agent.save(str(path))
    loaded = Policy.load(str(path))
    env = run_episode(generate_task(9000), loaded)
    assert env.done
