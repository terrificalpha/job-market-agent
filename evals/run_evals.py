"""Eval harness: runs fixed queries through the agent and scores them with deterministic checks
(tool order, tool errors, answer content, files written). No LLM judge, so results are reproducible.

Usage (from the project root):
    python -m evals.run_evals --model qwen2.5:7b
    python -m evals.run_evals --model llama3.2:3b --limit 5
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import agent.agent as agent_mod
from agent import tools
from agent.agent import run_agent

EVAL_DIR = Path(__file__).parent


def is_subsequence(expected: list, actual: list) -> bool:
    it = iter(actual)
    return all(t in it for t in expected)


def check_case(case: dict, result: dict) -> list[str]:
    """Return a list of failure reasons (empty list = pass)."""
    fails = []
    if result["status"] != "ok":
        fails.append(f"status={result['status']}")
    calls = result["tool_calls"]
    if case.get("expect_no_tools") and calls:
        fails.append(f"called tools when none were needed: {calls}")
    expected = case.get("expected_tools", [])
    if expected and not is_subsequence(expected, calls):
        fails.append(f"expected tool order {expected}, got {calls}")
    if result["tool_errors"] > case.get("max_tool_errors", 0):
        fails.append(f"{result['tool_errors']} tool errors")
    answer = result["answer"].lower()
    for needle in case.get("must_contain_all", []):
        if needle.lower() not in answer:
            fails.append(f"answer missing '{needle}'")
    any_of = case.get("must_contain_any", [])
    if any_of and not any(n.lower() in answer for n in any_of):
        fails.append(f"answer has none of {any_of[:4]}...")
    for needle in case.get("must_not_contain", []):
        if needle.lower() in answer:
            fails.append(f"answer contains forbidden '{needle}'")
    return fails


def run_case(case: dict, client=None) -> dict:
    report = case.get("file_saved")
    if report:
        (tools.REPORTS / report).unlink(missing_ok=True)  # make sure the file is freshly created
    try:
        result = run_agent(case["query"], client=client)
    except Exception as exc:  # a crashed run is a failed case, not a crashed eval
        result = {"status": f"crash: {exc!r}", "answer": "", "tool_calls": [],
                  "tool_errors": 0, "steps": 0, "elapsed_s": 0}
    fails = check_case(case, result)
    if report and not (tools.REPORTS / report).exists():
        fails.append(f"report {report} was not saved")
    return {"id": case["id"], "passed": not fails, "fails": fails, "steps": result["steps"],
            "elapsed_s": result["elapsed_s"], "tool_calls": result["tool_calls"],
            "trace": result.get("trace", []), "answer": result["answer"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=agent_mod.MODEL)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    args = ap.parse_args()
    agent_mod.MODEL = args.model

    cases = json.loads((EVAL_DIR / "cases.json").read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]

    print(f"Model: {args.model} | cases: {len(cases)}\n")
    results = []
    for case in cases:
        r = run_case(case)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']:<18} {r['elapsed_s']:>6}s  steps={r['steps']}", flush=True)
        for f in r["fails"]:
            print(f"        - {f}")
        if not r["passed"]:  # show exactly what the model sent to each tool and what came back
            for t in r["trace"]:
                print(f"        > {t['tool']}({json.dumps(t['input'])}) -> {t['output'][:150]}")
            print(f"        answer: {r['answer'][:200]!r}")

    passed = sum(r["passed"] for r in results)
    rate = 100 * passed / len(results)
    avg_t = sum(r["elapsed_s"] for r in results) / len(results)
    avg_s = sum(r["steps"] for r in results) / len(results)
    print(f"\nPass rate: {passed}/{len(results)} = {rate:.0f}%  | avg {avg_t:.1f}s, {avg_s:.1f} steps")

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = args.model.replace(":", "_").replace("/", "_")
    (out_dir / f"{safe}_{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = EVAL_DIR / "RESULTS.md"
    if not summary.exists():
        summary.write_text("| Date | Model | Cases | Pass rate | Avg time (s) | Avg steps |\n"
                           "|---|---|---|---|---|---|\n", encoding="utf-8")
    with summary.open("a", encoding="utf-8") as f:
        f.write(f"| {datetime.now():%Y-%m-%d} | {args.model} | {len(results)} | "
                f"{rate:.0f}% | {avg_t:.1f} | {avg_s:.1f} |\n")


if __name__ == "__main__":
    main()
