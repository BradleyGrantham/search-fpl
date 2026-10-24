"""Summarise benchmark runs: latency percentiles, and how often answers match a reference.

    uv run python bench/report.py bench/results/baseline.jsonl
    uv run python bench/report.py bench/results/candidate.jsonl --against bench/ground_truth.jsonl
    uv run python bench/report.py bench/results/baseline.jsonl --write-ground-truth

Ground truth is, per question, the answer (ordered player ids) the baseline gave most often,
with the SQL that produced it and how often the baseline agreed with itself. A candidate is
judged against that: same ids in the same order, same set in any order, or different.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
GROUND_TRUTH = BENCH / "ground_truth.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency_table(rows: list[dict]) -> str:
    ok = [r for r in rows if "total_ms" in r]
    lines = [f"{'':10}{'mean':>8}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}"]
    for field in ("total_ms", "model_ms", "sql_ms"):
        v = [r[field] for r in ok]
        cols = [statistics.mean(v)] + [percentile(v, p) for p in (50, 90, 95, 99)] + [max(v)]
        lines.append(f"{field:10}" + "".join(f"{c:8.1f}" if field == "sql_ms" else f"{c:8.0f}" for c in cols))
    return "\n".join(lines)


def ground_truth(rows: list[dict]) -> list[dict]:
    by_q = defaultdict(list)
    for r in rows:
        if r.get("ids") is not None:
            by_q[r["q"]].append(r)
    truth = []
    for q, runs in by_q.items():
        counts = Counter(tuple(r["ids"]) for r in runs)
        ids, n = counts.most_common(1)[0]
        example = next(r for r in runs if tuple(r["ids"]) == ids)
        truth.append({
            "q": q, "ids": list(ids), "stats": example["stats"], "sql": example["sql"],
            "agreement": round(n / len(runs), 2), "distinct_answers": len(counts), "runs": len(runs),
        })
    return truth


def compare(rows: list[dict], truth: list[dict]) -> Counter:
    ref = {t["q"]: t["ids"] for t in truth}
    verdicts = Counter()
    for r in rows:
        if r["q"] not in ref:
            continue
        if r.get("ids") is None:
            verdicts["error"] += 1
        elif r["ids"] == ref[r["q"]]:
            verdicts["exact"] += 1
        elif set(r["ids"]) == set(ref[r["q"]]):
            verdicts["same set, other order"] += 1
        else:
            verdicts["different"] += 1
    return verdicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--against", type=Path, help="ground truth to score this run against")
    parser.add_argument("--write-ground-truth", action="store_true")
    args = parser.parse_args()

    rows = load(args.run)
    ok = [r for r in rows if "total_ms" in r]
    print(f"{args.run.name}: {len(rows)} calls, {len({r['q'] for r in rows})} questions, "
          f"{len(rows) - len(ok)} errors, {sum(r.get('retried', False) for r in ok)} retried")
    cache_hits = sum(r["usage"]["cache_read"] > 0 for r in ok)
    print(f"prompt cache hit on {cache_hits}/{len(ok)}; "
          f"mean output tokens {statistics.mean(r['usage']['output'] for r in ok):.0f}\n")
    print(latency_table(rows))

    slowest = sorted(ok, key=lambda r: -r["total_ms"])[:5]
    print("\nslowest:\n" + "\n".join(f"  {r['total_ms']:7.0f} ms  {r['q']}" for r in slowest))

    truth = ground_truth(rows)
    if args.write_ground_truth:
        GROUND_TRUTH.write_text("".join(json.dumps(t) + "\n" for t in truth))
        print(f"\nwrote {len(truth)} questions to {GROUND_TRUTH}")
    agreement = [t["agreement"] for t in truth]
    if truth and truth[0]["runs"] > 1:
        print(f"\nself-agreement: mean {statistics.mean(agreement):.0%}, "
              f"{sum(a == 1 for a in agreement)}/{len(truth)} questions gave one answer every time")
        flaky = sorted(truth, key=lambda t: t["agreement"])[:8]
        print("least stable:\n" + "\n".join(
            f"  {t['agreement']:.0%} ({t['distinct_answers']} answers)  {t['q']}" for t in flaky if t["agreement"] < 1))

    if args.against:
        verdicts = compare(rows, load(args.against))
        total = sum(verdicts.values())
        print("\nvs ground truth:\n" + "\n".join(f"  {k:24}{v:5}  {v / total:.0%}" for k, v in verdicts.most_common()))


if __name__ == "__main__":
    main()
