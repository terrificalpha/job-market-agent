from types import SimpleNamespace as NS

from agent import tools
from evals.run_evals import check_case, is_subsequence, run_case


def ok(**kw):
    base = {"status": "ok", "answer": "", "tool_calls": [], "tool_errors": 0, "steps": 1, "elapsed_s": 0}
    return {**base, **kw}


def test_subsequence():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])


def test_check_passes_and_fails():
    case = {"expected_tools": ["search_jobs"], "must_contain_all": ["docker"], "must_not_contain": ["J0"]}
    assert check_case(case, ok(tool_calls=["search_jobs"], answer="You lack Docker")) == []
    assert check_case(case, ok(tool_calls=[], answer="You lack Docker"))        # wrong tools
    assert check_case(case, ok(tool_calls=["search_jobs"], answer="nothing"))   # missing content
    assert check_case(case, ok(tool_calls=["search_jobs"], answer="docker J001"))  # forbidden text


def test_no_tools_and_error_limits():
    assert check_case({"expect_no_tools": True}, ok(tool_calls=["search_jobs"]))
    assert check_case({}, ok(tool_errors=1))
    assert check_case({"max_tool_errors": 1}, ok(tool_errors=1)) == []


def test_run_case_checks_saved_report(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "REPORTS", tmp_path)
    call = lambda n, **a: NS(function=NS(name=n, arguments=a))
    script = [NS(message=NS(content="", tool_calls=[call("search_jobs", query="AI Engineer")])),
              NS(message=NS(content="", tool_calls=[call("save_report", filename="x.md", content="# r")])),
              NS(message=NS(content="docker done", tool_calls=[]))]

    class Fake:
        def chat(self, **kw):
            return script.pop(0)

    case = {"id": "t", "query": "q", "expected_tools": ["search_jobs", "save_report"],
            "must_contain_all": ["docker"], "file_saved": "x.md"}
    r = run_case(case, client=Fake())
    assert r["passed"], r["fails"]
    assert r["trace"][0]["tool"] == "search_jobs"
