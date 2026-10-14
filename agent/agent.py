"""The agent loop (Ollama version): call the model, run any tools it asks for,
feed results back, repeat until it answers without requesting a tool."""
import json
import os
import time
import uuid

from .logger import log_event
from .tools import TOOL_SCHEMAS, run_tool

MODEL = os.getenv("MODEL", "qwen2.5:7b")
MAX_STEPS = 8
LLM_RETRIES = 3
MAX_NUDGES = 2  # how many times to re-prompt after an empty model reply
NUDGE = ("Your last reply was empty. Continue the task: call the next required tool, or if all "
         "tools are finished, write the final reply in the required format.")

SYSTEM = (
    "You are a job-market analyst agent. If the request is not about jobs, skills or resumes, "
    "answer directly and call no tools. Otherwise work in this order: 1) call search_jobs, "
    "2) call compare_to_resume with the job ids returned, 3) call save_report with a short "
    "markdown report, 4) reply with a brief summary. Skip a step only if the request does not "
    "need it (a question about one job id needs only extract_skills). Keep calling tools until "
    "every part of the request is done; do not reply before then. Only state facts from tool "
    "results. If a tool returns an error, fix the input and try again once; if a job id does "
    "not exist, tell the user it was not found.\n\n"
    "FINAL REPLY FORMAT (only after all tools are finished). If you used search_jobs: first "
    "line 'Found N job(s).' with the number of jobs, then each job as "
    "'JOB_ID - Title - Company - Location', then your summary. If no jobs were found, say so."
)

# Ollama expects {"type": "function", "function": {name, description, parameters}}
OLLAMA_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOL_SCHEMAS
]


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import httpx
        import ollama
    except ImportError:
        return False
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return isinstance(exc, ollama.ResponseError) and getattr(exc, "status_code", 0) >= 500


def _call_llm(client, messages, run_id):
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            return client.chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS,
                               options={"temperature": 0, "num_ctx": 4096})
        except Exception as exc:
            if not _retryable(exc) or attempt == LLM_RETRIES:
                log_event(run_id, "llm_failed", attempt=attempt, error=repr(exc))
                raise
            wait = 2 ** attempt
            log_event(run_id, "llm_retry", attempt=attempt, wait_s=wait, error=repr(exc))
            time.sleep(wait)


def run_agent(user_query: str, client=None, max_steps: int = MAX_STEPS) -> dict:
    if client is None:
        import ollama
        client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                               timeout=180)  # local 7B models on a 4 GB GPU can be slow
    run_id = uuid.uuid4().hex[:8]
    log_event(run_id, "run_start", query=user_query, model=MODEL)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_query}]
    tool_calls = []
    tool_errors = 0
    nudges = 0
    trace = []  # what each tool was called with and returned (for eval debugging)
    failed_calls = set()  # (tool, args) pairs that already errored
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        resp = _call_llm(client, messages, run_id)
        msg = resp.message
        calls = msg.tool_calls or []
        log_event(run_id, "llm_response", step=step, n_tool_calls=len(calls))
        messages.append(msg)

        if not calls and not (msg.content or "").strip() and nudges < MAX_NUDGES:
            # small local models sometimes return an empty message after a tool result
            nudges += 1
            log_event(run_id, "empty_reply_nudge", step=step, nudge=nudges)
            messages.pop()  # drop the empty assistant turn so it doesn't pollute the history
            messages.append({"role": "user", "content": NUDGE})
            continue

        if not calls:
            log_event(run_id, "run_end", status="ok", steps=step)
            return {"status": "ok", "answer": msg.content or "", "steps": step,
                    "tool_calls": tool_calls, "tool_errors": tool_errors, "trace": trace,
                    "elapsed_s": round(time.perf_counter() - t0, 1)}

        for call in calls:
            name, args = call.function.name, dict(call.function.arguments or {})
            start = time.perf_counter()
            key = (name, json.dumps(args, sort_keys=True, default=str))
            if key in failed_calls:  # stop the model from retrying the exact same failing call
                output, is_error = ("Tool error: this exact call already failed. Do not repeat it; "
                                    "explain the problem to the user."), True
            else:
                try:
                    output, is_error = run_tool(name, args), False
                except Exception as exc:  # any tool failure goes back to the model, not a crash
                    output, is_error = f"Tool error: {exc}", True
                    failed_calls.add(key)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            tool_calls.append(name)
            tool_errors += is_error
            trace.append({"tool": name, "input": args, "output": output[:300], "error": is_error})
            log_event(run_id, "tool_call", step=step, tool=name, input=args,
                      is_error=is_error, latency_ms=latency_ms, output=output[:300])
            messages.append({"role": "tool", "tool_name": name, "content": output})

    log_event(run_id, "run_end", status="max_steps", steps=max_steps)
    return {"status": "max_steps", "answer": "Stopped: step limit reached.",
            "steps": max_steps, "tool_calls": tool_calls, "tool_errors": tool_errors, "trace": trace,
            "elapsed_s": round(time.perf_counter() - t0, 1)}
