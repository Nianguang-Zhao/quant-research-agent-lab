import json
from quantresearch.market import generate_task
from quantresearch import llm_agent
from quantresearch.sft_examples import build_examples


def test_llm_agent_can_make_only_whitelisted_tool_calls(monkeypatch):
    sequence = iter([{"action": "trend_z0", "reason": "test trend"},
                     {"action": "STOP", "reason": "budget saved"}])
    monkeypatch.setattr(llm_agent, "query_compatible_api", lambda *a, **kw: next(sequence))
    env, trace = llm_agent.run_llm_research(generate_task(777),
                                             url="http://localhost/unused", model="unit-test")
    assert env.done and env.steps == 1
    assert len(trace) == 2
    assert trace[0]["selection"] == "trend_z0"


def test_teacher_examples_are_bounded_training_data(tmp_path):
    dest = tmp_path / "sft.jsonl"
    build_examples(4, dest)
    items = [json.loads(line) for line in dest.read_text().splitlines()]
    assert len(items) == 4
    assert all(len(row["messages"]) == 3 for row in items)
    assert all("lockbox" not in json.dumps(row).lower() for row in items)
