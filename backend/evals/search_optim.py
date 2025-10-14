import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "_data"
DEFAULT_QUERIES_FILE = DATA_DIR / "queries.py"
SERVER_URL = (os.getenv("BACKEND_PUBLIC_URL") or "http://localhost:8000").rstrip("/")
load_dotenv(dotenv_path=REPO_ROOT / "evals" / ".env")
TOKEN = os.getenv("ACCESS_TOKEN")
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMTVkNTlkMzUtZWFlMy00Yzk5LTg3MTUtODZiNGJhNGU0MWMzIiwidXNlcm5hbWUiOiJ0ZXN0IiwiaWF0IjoxNzYwMzkwODAwLCJqdGkiOiIwZmMwZjg2NS02NWQzLTRlNDQtYmNmYy04ZWQyYWY1MGE2NjIifQ.vcNLNTMkwSLYTFMHziFD4txnia2T_xrHyQtfjSGNIAA"

def load_queries(path, subset=None):
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found at {path}")
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if subset is None:
            raw = [q for s in data.values() for q in s.get("query", [])]
        else:
            section = data.get(subset, {})
            raw = section.get("query", []) if isinstance(section, dict) else section
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]

    # normalize to list[dict]
    out = []
    for x in raw:
        if isinstance(x, dict) and "query" in x:
            out.append(x)
        elif isinstance(x, str):
            out.append({"query": x})
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run agentic search queries and save the outputs.")
    parser.add_argument("--input", type=Path, default="backend/evals/_data/queries.json")
    parser.add_argument("--output", type=Path, default="backend/evals/_data")
    return parser.parse_args()


def _consume_sse(resp):
    """Yield JSON payloads from a text/event-stream response.

    Lines are of the form: `data: {json}`. We ignore other fields/events.
    """
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if not data_str:
            continue
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        yield payload


def run_search(query, mode):
    """Call the search endpoint, handling SSE or JSON.

    Returns a dict with `results` and `agent` keys to match historical callers.
    Prints status updates in real-time as they stream.
    """
    # Prefer SSE (server streams agent events)
    headers = {
        "Accept": "text/event-stream, application/json",
    }
    headers["Authorization"] = f"Bearer {TOKEN}"

    with requests.get(
        f"{SERVER_URL}/items/search",
        headers=headers,
        params={
            "query": query,
            "mode": mode
        },
        timeout=120,
        stream=True,
    ) as resp:
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()
        # If server returned JSON directly, pass it through
        if "application/json" in content_type:
            try:
                payload = resp.json()
                # Normalise common error payloads for easier debugging downstream
                if isinstance(payload, dict) and "results" in payload:
                    return payload
                return {"results": [], "agent": {"backend": "json", "state": "error", "reason": payload}}
            except Exception:
                pass

        # Otherwise, consume the SSE stream and build a payload
        events = []
        final_items = []
        final_state = "done"
        final_reason = None
        try:
            for ev in _consume_sse(resp):
                events.append(ev)
                # Print status events in real-time with reasoning
                if ev.get("kind") == "status":
                    msg = ev.get("message", "").strip()
                    reasoning = ev.get("reasoning", "").strip()
                    print(f"  → {msg}", flush=True)
                    if reasoning:
                        print(f"     {reasoning}", flush=True)
                elif ev.get("kind") == "final":
                    if isinstance(ev.get("data"), list):
                        final_items = ev["data"]
                    msg = str(ev.get("message") or "")
                    if msg == "no-results":
                        final_state = "no-results"
                        final_reason = ev.get("reason") or "no-justified-items"
                    break
        except Exception as exc:
            final_state = "error"
            final_reason = f"sse-consume-failed: {exc.__class__.__name__}"

        # If we received no events at all, surface a clearer reason
        if not events and not final_items and final_state == "done":
            final_state = "no-events"
            final_reason = "empty-stream"

        agent_meta = {
            "steps": [e for e in events if e.get("kind") == "status"],
            "backend": "stream",
            "state": final_state,
        }
        if final_reason:
            agent_meta["reason"] = final_reason

        return {"results": final_items, "agent": agent_meta}


def summarise_results(results):
    """Print clean summary with titles and justifications."""
    if not results:
        print("  No results\n", flush=True)
        return

    print(f"\n  Found {len(results)} result(s):", flush=True)
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title") or "No title"
        justification = item.get("justification", "")
        # Prefer combined_score if provided by backend, else fallback to score
        eff_score = item.get("combined_score")
        if not isinstance(eff_score, (int, float)):
            eff_score = item.get("score")

        score_str = f" [{eff_score:.3f}]" if isinstance(eff_score, (int, float)) else ""
        print(f"  {idx}. {title}{score_str}", flush=True)
        if justification:
            print(f"     → {justification}", flush=True)
    print("", flush=True)


def build_output_path(cli_path):
    if cli_path:
        p = Path(cli_path)
        # If a directory was provided, create a timestamped file inside it
        if p.exists() and p.is_dir():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            return p / f"search_optim_{timestamp}.json"
        # Otherwise, treat it as a file path (parent may be created later)
        return p
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"search_optim_{timestamp}.json"


def main():
    args = parse_args()
    queries = load_queries(args.input, subset="ACTIVE")

    run_records = []
    for query in queries:
        print(f"\n=== {query['query']} ===", flush=True)
        payload = run_search(query["query"], 'agentic')
        results = payload.get("results")
        summarise_results(results)
        run_records.append(
            {
                "query": query,
                "payload": payload,
            }
        )

    output_path = build_output_path(args.output)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": run_records,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"Saved run data to {output_path}", flush=True)


if __name__ == "__main__":
    main()
