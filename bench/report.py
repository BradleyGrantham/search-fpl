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
import random
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


def bootstrap_delta(pairs: list[tuple[float, float]], p: float, n: int = 2000) -> tuple[float, float, float]:
    """Change in the p-th percentile (b - a) over paired calls, with a 95% bootstrap interval."""
    rng = random.Random(0)
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    point = percentile(b, p) - percentile(a, p)
    draws = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws.append(percentile([y for _, y in sample], p) - percentile([x for x, _ in sample], p))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def modal_answers(rows: list[dict]) -> dict:
    by_q = defaultdict(Counter)
    sql = {}
    for r in rows:
        if r.get("ids") is not None:
            by_q[r["q"]][tuple(r["ids"])] += 1
            sql.setdefault((r["q"], tuple(r["ids"])), r["sql"])
    return {q: (list(c.most_common(1)[0][0]), sql[(q, c.most_common(1)[0][0])]) for q, c in by_q.items()}


def compare_variants(rows: list[dict], truth: list[dict], ref: str, diffs: bool, json_out: Path | None):
    by_v = defaultdict(list)
    for r in rows:
        by_v[r.get("variant", "baseline")].append(r)
    ref_rows = {(r["q"], r["rep"]): r for r in by_v[ref] if "total_ms" in r}
    ref_modal = modal_answers(by_v[ref])
    tq = {t["q"]: t for t in truth}
    summary = []
    print(f"{'variant':12}{'in tok':>7}{'out tok':>8}{'retry':>6}{'p50':>6}{'p90':>6}{'p99':>6}{'mean':>6}"
          f"   {'Δp50 [95% CI]':>18}   {'Δp90 [95% CI]':>18}   {'exact':>6}{'set':>5}{'diff':>5}")
    for name, vr in by_v.items():
        ok = [r for r in vr if "total_ms" in r]
        t = [r["total_ms"] for r in ok]
        tin = statistics.mean(r["usage"]["input"] + r["usage"]["cache_read"] + r["usage"]["cache_write"] for r in ok)
        tout = statistics.mean(r["usage"]["output"] for r in ok)
        pairs = [(ref_rows[(r["q"], r["rep"])]["total_ms"], r["total_ms"]) for r in ok if (r["q"], r["rep"]) in ref_rows]
        d50, d90 = bootstrap_delta(pairs, 50), bootstrap_delta(pairs, 90)
        v = compare(vr, truth)
        total = sum(v.values()) or 1
        row = {
            "variant": name, "calls": len(vr), "errors": len(vr) - len(ok), "retried": sum(r["retried"] for r in ok),
            "cache_hits": sum(r["usage"]["cache_read"] > 0 for r in ok),
            "input_tokens": round(tin), "output_tokens": round(tout, 1),
            "p50": round(percentile(t, 50)), "p90": round(percentile(t, 90)), "p95": round(percentile(t, 95)),
            "p99": round(percentile(t, 99)), "mean": round(statistics.mean(t)),
            "d_p50": round(d50[0]), "d_p50_lo": round(d50[1]), "d_p50_hi": round(d50[2]),
            "d_p90": round(d90[0]), "d_p90_lo": round(d90[1]), "d_p90_hi": round(d90[2]),
            "exact": round(v["exact"] / total, 3), "same_set": round(v["same set, other order"] / total, 3),
            "different": round((v["different"] + v["error"]) / total, 3),
        }
        summary.append(row)
        print(f"{name:12}{row['input_tokens']:7}{row['output_tokens']:8.1f}{row['retried']:6}{row['p50']:6}{row['p90']:6}"
              f"{row['p99']:6}{row['mean']:6}   {d50[0]:+5.0f} [{d50[1]:+5.0f},{d50[2]:+5.0f}]"
              f"   {d90[0]:+5.0f} [{d90[1]:+5.0f},{d90[2]:+5.0f}]   {row['exact']:6.0%}{row['same_set']:5.0%}{row['different']:5.0%}")

    retried = [(r["variant"], r["q"], r["first_attempt"]) for r in rows if r.get("first_attempt")]
    if retried:
        print("\nfirst attempts that failed (variant, question, error):")
        for v, q, first in sorted({(v, q, f["error"]) for v, q, f in retried}):
            print(f"  {v:10} {q:38} {first}")

    if diffs:
        for name, vr in by_v.items():
            if name == ref:
                continue
            modal = modal_answers(vr)
            changed = [q for q in modal if q in ref_modal and set(modal[q][0]) != set(ref_modal[q][0])]
            print(f"\n=== {name}: {len(changed)} questions whose usual answer differs from {ref}'s ===")
            for q in changed:
                ref_ok = set(ref_modal[q][0]) == set(tq[q]["ids"]) if q in tq else None
                print(f"\n  {q}   ({ref} matches ground truth: {ref_ok})")
                print(f"    {ref:9} {len(ref_modal[q][0]):2}  {ref_modal[q][1]}")
                print(f"    {name:9} {len(modal[q][0]):2}  {modal[q][1]}")
    streamed = [r for r in rows if r.get("usage", {}).get("ttft_ms") is not None and not r.get("retried")]
    if streamed:
        print(f"\nstreamed, first attempts only: {'variant':12}{'ttft p50':>9}{'gen p50':>8}{'gen ms/token':>13}")
        for row in summary:
            vr = [r for r in streamed if r["variant"] == row["variant"]]
            if not vr:
                continue
            ttft = percentile([r["usage"]["ttft_ms"] for r in vr], 50)
            gen = percentile([r["usage"]["gen_ms"] for r in vr], 50)
            per_tok = statistics.median(r["usage"]["gen_ms"] / max(r["usage"]["output"], 1) for r in vr)
            row.update(ttft_p50=round(ttft), gen_p50=round(gen), gen_ms_per_token=round(per_tok, 2))
            print(f"{'':30}{row['variant']:12}{ttft:9.0f}{gen:8.0f}{per_tok:13.2f}")
        pairs = defaultdict(dict)
        for r in streamed:
            pairs[(r["q"], r["rep"])][r["variant"]] = r["usage"]
        for row in summary:
            if row["variant"] == ref:
                continue
            d = [(p[ref]["gen_ms"], p[row["variant"]]["gen_ms"]) for p in pairs.values() if ref in p and row["variant"] in p]
            t = [(p[ref]["ttft_ms"], p[row["variant"]]["ttft_ms"]) for p in pairs.values() if ref in p and row["variant"] in p]
            if d:
                g, tt = bootstrap_delta(d, 50), bootstrap_delta(t, 50)
                row.update(d_gen=round(g[0]), d_gen_lo=round(g[1]), d_gen_hi=round(g[2]), d_ttft=round(tt[0]))
                print(f"{'':30}{row['variant']:12} Δgen p50 {g[0]:+.0f} ms [{g[1]:+.0f},{g[2]:+.0f}]   Δttft p50 {tt[0]:+.0f} [{tt[1]:+.0f},{tt[2]:+.0f}]")
    if json_out:
        json_out.write_text(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--against", type=Path, help="ground truth to score this run against")
    parser.add_argument("--write-ground-truth", action="store_true")
    parser.add_argument("--ref", default="baseline", help="variant the others are compared with")
    parser.add_argument("--diffs", action="store_true", help="list questions whose answer changed")
    parser.add_argument("--json", type=Path, help="write the per-variant summary here")
    args = parser.parse_args()

    rows = load(args.run)
    if len({r.get("variant", "baseline") for r in rows}) > 1:
        compare_variants(rows, load(args.against or GROUND_TRUTH), args.ref, args.diffs, args.json)
        return
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
