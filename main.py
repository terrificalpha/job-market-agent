import sys

from dotenv import load_dotenv

load_dotenv()

from agent.agent import run_agent  # noqa: E402  (after load_dotenv so MODEL env is read)

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or (
        "Find AI Engineer roles in Hyderabad, work out the most-asked skills versus mine, "
        "and save a report called hyderabad_ai_gap.md"
    )
    result = run_agent(query)
    print(result["answer"])
    print(f"\n[{result['status']}] steps={result['steps']} tools={result['tool_calls']}")
