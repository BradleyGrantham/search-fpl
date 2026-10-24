"""Benchmark the /ask pipeline: question -> Claude -> SQL -> player ids.

Runs the same code the Worker runs (llm.generate_sql, nl.clean_sql, nl.run_sql, one retry on
a SQL error) against a frozen copy of the database, so answers are comparable between runs.
Every call is written to a JSONL file; `report.py` turns that into latency percentiles and
answer agreement.

    uv run python bench/run.py --reps 5 --label baseline
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import llm  # noqa: E402
import nl  # noqa: E402

BENCH = Path(__file__).resolve().parent
SNAPSHOT = BENCH / "snapshot.db"  # frozen copy of fpl.db; answers are only comparable on one snapshot


def load_questions(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Set ANTHROPIC_API_KEY or put it in .env")


async def ask_once(key: str, con, prompt: str, question: str) -> dict:
    """Mirrors entry.ask, with a timer around each step."""
    t0 = time.perf_counter()
    text, usage = await llm.generate_sql(key, prompt, question)
    sql = nl.clean_sql(text)
    t1 = time.perf_counter()
    model_ms, sql_ms, retried, error = (t1 - t0) * 1000, 0.0, False, None
    try:
        ids, stats = nl.run_sql(con, sql)
        sql_ms = (time.perf_counter() - t1) * 1000
    except (nl.sqlite3.Error, ValueError) as err:
        retried = True
        t2 = time.perf_counter()
        text, retry_usage = await llm.generate_sql(key, prompt, question, failed_sql=sql, error=str(err))
        sql = nl.clean_sql(text)
        usage = {k: usage[k] + retry_usage[k] for k in usage}
        t3 = time.perf_counter()
        model_ms += (t3 - t2) * 1000
        try:
            ids, stats = nl.run_sql(con, sql)
        except (nl.sqlite3.Error, ValueError) as err2:
            ids, stats, error = [], [], str(err2)
        sql_ms = (time.perf_counter() - t3) * 1000
    return {
        "q": question, "sql": sql, "ids": ids, "stats": stats, "stat_label": nl.stat_label(sql),
        "retried": retried, "error": error, "usage": usage,
        "model_ms": round(model_ms, 1), "sql_ms": round(sql_ms, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=BENCH / "questions.txt")
    parser.add_argument("--reps", type=int, default=5, help="passes over the whole question set")
    parser.add_argument("--label", default="run")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = parser.parse_args()

    if not SNAPSHOT.exists():
        shutil.copy(ROOT / "fpl.db", SNAPSHOT)
        print(f"Froze {ROOT / 'fpl.db'} as {SNAPSHOT}")
    con = nl.open_read_only(SNAPSHOT.read_bytes())
    prompt = nl.build_system_prompt(con)
    questions = load_questions(args.questions)[: args.limit]
    key = api_key()

    out = BENCH / "results" / f"{args.label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    print(f"Warming the prompt cache: {await llm.warm_up(key, prompt)}")
    print(f"{len(questions)} questions x {args.reps} reps -> {out}")

    with out.open("w") as f:
        # Sequential on purpose: one visitor at a time, as the site sees it.
        for rep in range(args.reps):
            for i, q in enumerate(questions):
                started = datetime.now(timezone.utc).isoformat()
                try:
                    row = await ask_once(key, con, prompt, q)
                except Exception as err:  # timeouts, refusals, API errors: count them, keep going
                    row = {"q": q, "error": f"{type(err).__name__}: {err}", "ids": None}
                row.update(rep=rep, ts=started, model=llm.MODEL)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"rep {rep} {i + 1:3}/{len(questions)} {row.get('total_ms', '-'):>8} ms  {q}")


if __name__ == "__main__":
    asyncio.run(main())
