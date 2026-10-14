import json
from types import SimpleNamespace as NS

from agent import tools
from agent.agent import run_agent


def call(name, **args):
    return NS(function=NS(name=name, arguments=args))


def reply(content="", *calls):
    return NS(message=NS(content=content, tool_calls=list(calls)))


class FakeClient:
    """Plays back scripted Ollama responses so the loop can be tested without a model."""
    def __init__(self, script):
        self.script = list(script)

    def chat(self, **kwargs):
        return self.script.pop(0)


def test_search_and_compare_tools():
    found = json.loads(tools.search_jobs("AI Engineer", "Hyderabad"))
    assert found["count"] >= 1
    ids = [j["id"] for j in found["jobs"]]
    gaps = json.loads(tools.compare_to_resume(ids))["gaps"]
    assert any(g["skill"] == "docker" for g in gaps)


def test_compare_accepts_string_ids():
    out = json.loads(tools.compare_to_resume("J001, J010"))
    assert out["jobs_analyzed"] == 2


def test_agent_chains_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "REPORTS", tmp_path)
    client = FakeClient([
        reply("", call("search_jobs", query="AI Engineer")),
        reply("", call("compare_to_resume", job_ids=["J001", "J010"])),
        reply("", call("save_report", filename="r.md", content="# Gaps")),
        reply("Done."),
    ])
    out = run_agent("go", client=client)
    assert out["status"] == "ok"
    assert out["tool_calls"] == ["search_jobs", "compare_to_resume", "save_report"]
    assert (tmp_path / "r.md").exists()


def test_tool_error_is_recoverable():
    client = FakeClient([
        reply("", call("extract_skills", job_id="NOPE")),
        reply("", call("extract_skills", job_id="J001")),
        reply("Recovered."),
    ])
    out = run_agent("go", client=client)
    assert out["status"] == "ok" and out["steps"] == 3


def test_unknown_tool_is_recoverable():
    client = FakeClient([reply("", call("make_coffee")), reply("Ok.")])
    assert run_agent("go", client=client)["status"] == "ok"


def test_max_steps_guard():
    loop = [reply("", call("search_jobs", query="AI")) for _ in range(10)]
    out = run_agent("go", client=FakeClient(loop), max_steps=3)
    assert out["status"] == "max_steps"


def test_duplicate_failed_call_is_blocked(monkeypatch):
    import agent.agent as A
    calls = []
    real = A.run_tool

    def counting(name, args):
        calls.append(name)
        return real(name, args)

    monkeypatch.setattr(A, "run_tool", counting)
    client = FakeClient([
        reply("", call("extract_skills", job_id="NOPE")),
        reply("", call("extract_skills", job_id="NOPE")),
        reply("Not found."),
    ])
    out = run_agent("go", client=client)
    assert calls == ["extract_skills"]      # second identical call never reached the tool
    assert out["tool_errors"] == 2 and out["status"] == "ok"


def test_search_tolerates_location_in_query():
    ids = lambda q, l="": [j["id"] for j in json.loads(tools.search_jobs(q, l))["jobs"]]
    assert ids("Remote Data Scientist") == ["J009"]
    assert ids("Machine Learning Engineer Bangalore") == ["J002"]
    assert ids("AI Engineer jobs in Pune") == ["J010"]
    assert ids("AI Engineer", "Hyderabad") == ["J001", "J003"]
    assert ids("Quantum Engineer", "Hyderabad") == []


def test_empty_reply_is_nudged_and_chain_continues():
    client = FakeClient([
        reply("", call("search_jobs", query="AI Engineer")),
        reply(""),                                          # model goes silent after the tool result
        reply("", call("compare_to_resume", job_ids=["J001"])),
        reply("Done."),
    ])
    out = run_agent("go", client=client)
    assert out["status"] == "ok" and out["answer"] == "Done."
    assert out["tool_calls"] == ["search_jobs", "compare_to_resume"]


def test_repeated_empty_replies_terminate():
    client = FakeClient([reply(""), reply(""), reply("")])
    out = run_agent("go", client=client)
    assert out["status"] == "ok" and out["answer"] == ""    # gives up after MAX_NUDGES, no infinite loop
