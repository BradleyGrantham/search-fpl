"""Benchmark the /ask pipeline: question -> Claude -> SQL -> player ids.

Runs the same steps the Worker runs (the Claude call, nl.clean_sql, running the SQL, one retry
on a SQL error) against a frozen copy of the database, so answers are comparable between runs.
Every call is written to a JSONL file; `report.py` turns that into latency percentiles and
answer agreement.

    uv run python bench/run.py --reps 5 --label worker
    uv run python bench/run.py --reps 3 --label vs-old --variants baseline,worker

With several variants (see variants.py), each question is asked of every variant in a random
order before moving on, so drift in API speed hits all of them equally.
"""

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import llm  # noqa: E402
import nl  # noqa: E402
import legacy  # noqa: E402
from variants import VARIANTS, open_db  # noqa: E402

BENCH = Path(__file__).resolve().parent
SNAPSHOT = BENCH / "snapshot.db"  # frozen copy of fpl.db; answers are only comparable on one snapshot
STREAM = False  # --stream: time to first token and generation time, recorded separately


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


async def generate(key: str, variant, prompt: str, messages: list[dict], max_tokens: int = legacy.MAX_TOKENS):
    """llm.generate_sql, with the variant's caching and API settings (and optional hedging)."""
    params = dict(model=llm.MODEL, max_tokens=max_tokens, thinking=llm.THINKING,
                  system=llm._system(prompt) if variant.cache else prompt, messages=messages)
    params.update({k: v for k, v in variant.api.items() if not (max_tokens == 0 and k == "max_tokens")})

    async def call():
        return await llm._get_client(key).messages.create(**params)

    hedges, timing = 0, {}
    if STREAM and max_tokens:
        # Same request, streamed, to split time to first token from generation time.
        t0 = time.perf_counter()
        async with llm._get_client(key).messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "content_block_delta" and "ttft_ms" not in timing:
                    timing["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            response = await stream.get_final_message()
        timing["gen_ms"] = round((time.perf_counter() - t0) * 1000 - timing.get("ttft_ms", 0), 1)
    elif variant.hedge_ms and max_tokens:
        first = asyncio.create_task(call())
        done, _ = await asyncio.wait({first}, timeout=variant.hedge_ms / 1000)
        if done:
            response = first.result()
        else:
            hedges = 1
            second = asyncio.create_task(call())
            done, pending = await asyncio.wait({first, second}, return_when=asyncio.FIRST_COMPLETED)
            winner = done.pop()
            if winner.exception() and pending:  # one failed: wait for the other
                winner = pending.pop()
                await winner
            for task in pending:
                task.cancel()
            response = winner.result()
    else:
        response = await call()
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request ({response.stop_details})")
    text = next((block.text for block in response.content if block.type == "text"), "")
    return text, {**llm._usage(response), "hedges": hedges, **timing}


async def ask_once(key: str, variant, con, prompt: str, question: str) -> dict:
    """Mirrors entry.ask, with a timer around each step."""

    async def ask_model(failed_sql=None, error=None):
        if variant.via_llm and not STREAM:  # the Worker's own call, stop-sequence guard included
            return await llm.generate_sql(key, prompt, question, failed_sql, error)
        messages = [{"role": "user", "content": question}]
        if failed_sql:
            messages += [{"role": "assistant", "content": failed_sql},
                         {"role": "user", "content": f"That failed with: {error}\nFix it. Output only SQL."}]
        return await generate(key, variant, prompt, messages)

    t0 = time.perf_counter()
    text, usage = await ask_model()
    sql = nl.clean_sql(text)
    t1 = time.perf_counter()
    model_ms, sql_ms, retried, error, first = (t1 - t0) * 1000, 0.0, False, None, None
    try:
        is_answer = variant.is_answer or (lambda s: re.match(r"^\s*(SELECT|WITH)\b", s, re.IGNORECASE))
        if variant.prose_is_empty and not is_answer(sql):
            ids, stats, label = [], [], None
        else:
            ids, stats, label = variant.run(con, sql, variant.labels)
        sql_ms = (time.perf_counter() - t1) * 1000
    except (nl.sqlite3.Error, ValueError) as err:
        retried, first = True, {"sql": sql, "error": str(err)}
        t2 = time.perf_counter()
        text, retry_usage = await ask_model(sql, str(err))
        sql = nl.clean_sql(text)
        usage = {k: usage[k] + retry_usage[k] for k in usage}
        t3 = time.perf_counter()
        model_ms += (t3 - t2) * 1000
        try:
            ids, stats, label = variant.run(con, sql, variant.labels)
        except (nl.sqlite3.Error, ValueError) as err2:
            ids, stats, label, error = [], [], None, str(err2)
        sql_ms = (time.perf_counter() - t3) * 1000
    return {
        "q": question, "sql": sql, "ids": ids, "stats": stats, "stat_label": label,
        "retried": retried, "first_attempt": first, "error": error, "usage": usage,
        "model_ms": round(model_ms, 1), "sql_ms": round(sql_ms, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=BENCH / "questions.txt")
    parser.add_argument("--reps", type=int, default=5, help="passes over the whole question set")
    parser.add_argument("--label", default="run")
    parser.add_argument("--variants", default="worker", help="comma-separated names from variants.py")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream", action="store_true", help="stream responses to time the first token")
    args = parser.parse_args()
    global STREAM
    STREAM = args.stream

    if not SNAPSHOT.exists():
        shutil.copy(ROOT / "fpl.db", SNAPSHOT)
        print(f"Froze {ROOT / 'fpl.db'} as {SNAPSHOT}")
    db = SNAPSHOT.read_bytes()
    arms = []
    for name in args.variants.split(","):
        v = VARIANTS[name]
        con = v.open(db) if v.open else open_db(db, v.setup)
        arms.append((v, con, v.prompt(con)))
    questions = load_questions(args.questions)[: args.limit]
    key = api_key()
    rng = random.Random(args.seed)

    out = BENCH / "results" / f"{args.label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    for v, _, prompt in arms:
        if v.cache:  # write each prompt to the cache once, as the Worker's refresh does
            _, usage = await generate(key, v, prompt, [{"role": "user", "content": "warmup"}], max_tokens=0)
            print(f"warmed {v.name}: {usage}")
    print(f"{len(questions)} questions x {args.reps} reps x {len(arms)} variants -> {out}", flush=True)

    with out.open("w") as f:
        # Sequential on purpose: one visitor at a time, as the site sees it.
        for rep in range(args.reps):
            for i, q in enumerate(questions):
                for v, con, prompt in rng.sample(arms, len(arms)):
                    started = datetime.now(timezone.utc).isoformat()
                    try:
                        row = await ask_once(key, v, con, prompt, q)
                    except Exception as err:  # timeouts, refusals, API errors: count them, keep going
                        row = {"q": q, "error": f"{type(err).__name__}: {err}", "ids": None}
                    row.update(rep=rep, ts=started, model=llm.MODEL, variant=v.name)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                print(f"rep {rep} {i + 1:3}/{len(questions)}  {q}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
