"""Tools the agent can call. Each tool takes plain arguments and returns a JSON string.
Bad input raises ToolError; the agent loop turns that into an error result the model can recover from."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SKILL_VOCAB = [
    "python", "sql", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow",
    "fastapi", "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "git",
    "mlops", "airflow", "spark", "rag", "llm", "langchain", "langgraph",
    "vector database", "prompt engineering", "machine learning", "deep learning",
    "nlp", "computer vision", "statistics", "power bi", "tableau",
]


class ToolError(Exception):
    pass


def _load_jobs() -> list[dict]:
    return json.loads((DATA / "jobs.json").read_text(encoding="utf-8"))


def _has_word(term: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text.lower()) is not None


def _find_skills(text: str) -> list[str]:
    return [s for s in SKILL_VOCAB if _has_word(s, text)]


STOPWORDS = {"job", "jobs", "role", "roles", "position", "positions", "opening", "openings",
             "in", "for", "the", "a", "an", "at", "of", "near"}
CITY_ALIASES = {"bangalore": "bengaluru", "bengalore": "bengaluru"}


def _norm_terms(text: str) -> list[str]:
    words = [CITY_ALIASES.get(w, w) for w in text.lower().replace(",", " ").split()]
    return [w for w in words if w not in STOPWORDS]


def search_jobs(query: str, location: str = "") -> str:
    """All keywords must match the title, description or location, so it works whether the model
    sends 'Data Scientist' + location='Remote' or just 'Remote Data Scientist'."""
    terms = _norm_terms(query)
    if not terms:
        raise ToolError("query must contain at least one keyword")
    loc = " ".join(_norm_terms(location))
    results = []
    for job in _load_jobs():
        blob = f"{job['title']} {job['description']} {job['location']}"
        if not all(_has_word(t, blob) for t in terms):
            continue
        if loc and loc not in job["location"].lower():
            continue
        results.append({k: job[k] for k in ("id", "title", "company", "location")})
    return json.dumps({"count": len(results), "jobs": results})


def extract_skills(job_id: str) -> str:
    for job in _load_jobs():
        if job["id"] == job_id:
            return json.dumps({"job_id": job_id, "skills": _find_skills(job["description"])})
    raise ToolError(f"job_id '{job_id}' was not found (job does not exist). "
                    f"Do not retry it; tell the user it was not found.")


def compare_to_resume(job_ids: list[str]) -> str:
    if isinstance(job_ids, str):  # small models sometimes send "J001, J002" instead of a list
        job_ids = [j.strip() for j in job_ids.strip("[]").replace('"', "").split(",") if j.strip()]
    if not job_ids:
        raise ToolError("job_ids must not be empty")
    jobs = {j["id"]: j for j in _load_jobs()}
    unknown = [i for i in job_ids if i not in jobs]
    if unknown:
        raise ToolError(f"job_ids {unknown} were not found. Use only ids returned by search_jobs.")
    mine = set(json.loads((DATA / "my_skills.json").read_text(encoding="utf-8")))
    demand: dict[str, int] = {}
    for i in job_ids:
        for s in _find_skills(jobs[i]["description"]):
            demand[s] = demand.get(s, 0) + 1
    ranked = sorted(demand.items(), key=lambda kv: -kv[1])
    return json.dumps({
        "jobs_analyzed": len(job_ids),
        "have": [{"skill": s, "jobs": n} for s, n in ranked if s in mine],
        "gaps": [{"skill": s, "jobs": n} for s, n in ranked if s not in mine],
    })


def save_report(filename: str, content: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).name)
    if not name.endswith(".md"):
        name += ".md"
    if not content.strip():
        raise ToolError("content must not be empty")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(content, encoding="utf-8")
    return json.dumps({"saved": f"reports/{name}"})


TOOLS = {
    "search_jobs": search_jobs,
    "extract_skills": extract_skills,
    "compare_to_resume": compare_to_resume,
    "save_report": save_report,
}

TOOL_SCHEMAS = [
    {"name": "search_jobs",
     "description": "Search job listings by keywords (all words must match) and optional location. Returns job ids, titles, companies, locations.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "e.g. 'AI Engineer'"},
         "location": {"type": "string", "description": "optional city, e.g. 'Hyderabad'"}},
         "required": ["query"]}},
    {"name": "extract_skills",
     "description": "List the technical skills required by one job.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
    {"name": "compare_to_resume",
     "description": "Given job ids, rank the skills employers ask for and split them into skills the user has vs gaps.",
     "input_schema": {"type": "object", "properties": {
         "job_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["job_ids"]}},
    {"name": "save_report",
     "description": "Save a markdown report to disk. Use as the final step.",
     "input_schema": {"type": "object", "properties": {
         "filename": {"type": "string"}, "content": {"type": "string"}},
         "required": ["filename", "content"]}},
]


def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        raise ToolError(f"unknown tool: {name}")
    try:
        return TOOLS[name](**args)
    except TypeError as e:
        raise ToolError(f"bad arguments for {name}: {e}")
