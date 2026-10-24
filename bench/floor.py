"""How fast can Haiku answer at all? Time to first token for a one-word reply, by prompt size.

Each round asks every arm once, in random order: system prompts from ~10 to ~11k tokens, with
and without prompt caching, all told to reply only "yes". A `GET /v1/models` call in the same
round measures the network and API overhead with no model in the way.

    uv run python bench/floor.py --rounds 40
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent / "src"))

import legacy  # noqa: E402
import llm  # noqa: E402
from run import api_key  # noqa: E402

INSTRUCTION = "Whatever the user says, reply with only the word yes. The text below is reference only.\n\n"
PAD = legacy.RULES.format(max_rows=10) + legacy.DATA_DICTIONARY + legacy.EXAMPLES.format(next_gw=8)  # ~1k tokens of the old prompt


def arms():
    out = [("tiny", "Reply with only the word yes.", False)]
    for copies in (0.5, 1, 2, 4, 8):
        text = INSTRUCTION + (PAD * int(copies + 0.99))[: int(len(PAD) * copies)]
        out.append((f"{copies:g}x uncached", text, False))
        if copies >= 1:  # under ~512 tokens the API doesn't cache
            out.append((f"{copies:g}x cached", text, True))
    return out


async def ttft(client, system, cache):
    t0 = time.perf_counter()
    first = None
    async with client.messages.stream(
        model=llm.MODEL, max_tokens=5, thinking=llm.THINKING,
        system=[{"type": "text", "text": system, **({"cache_control": {"type": "ephemeral", "ttl": "1h"}} if cache else {})}],
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and first is None:
                first = time.perf_counter()
        msg = await stream.get_final_message()
    u = msg.usage
    return {"ttft_ms": round(((first or time.perf_counter()) - t0) * 1000, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "input": u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
            "cache_read": u.cache_read_input_tokens or 0, "output": u.output_tokens}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=40)
    args = parser.parse_args()
    client = llm._get_client(api_key())
    tests = arms()
    for name, text, cache in tests:  # warm caches (and connections)
        await ttft(client, text, cache)
    out = BENCH / "results" / "floor.jsonl"
    out.parent.mkdir(exist_ok=True)
    rng = random.Random(0)
    with out.open("w") as f:
        for i in range(args.rounds):
            for name, text, cache in rng.sample(tests + [("GET /v1/models", None, None)], len(tests) + 1):
                if text is None:
                    t0 = time.perf_counter()
                    await client.models.retrieve(llm.MODEL)
                    row = {"ttft_ms": round((time.perf_counter() - t0) * 1000, 1), "input": 0}
                else:
                    row = await ttft(client, text, cache)
                f.write(json.dumps({"arm": name, "round": i, **row}) + "\n")
            print(f"round {i + 1}/{args.rounds}", flush=True)

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    print(f"\n{'arm':16}{'input tok':>10}{'cache hits':>11}{'ttft p50':>9}{'p90':>7}")
    for name in [t[0] for t in tests] + ["GET /v1/models"]:
        r = [x for x in rows if x["arm"] == name]
        t = sorted(x["ttft_ms"] for x in r)
        hits = sum(x.get("cache_read", 0) > 0 for x in r)
        print(f"{name:16}{statistics.median(x['input'] for x in r):10.0f}{hits:>7}/{len(r):<3}"
              f"{statistics.median(t):9.0f}{t[int(0.9 * (len(t) - 1))]:7.0f}")


if __name__ == "__main__":
    asyncio.run(main())
