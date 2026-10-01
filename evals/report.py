"""Score table for one or more result files.

    python -m evals.report evals/results/baseline.jsonl [evals/results/other.jsonl ...]

pass@k: chance that at least one of k attempts passes (can the agent do it at all?).
pass^k: chance that all k attempts pass (can you rely on it?).
Both are computed with the unbiased estimator over the n trials that were run.
"""
import json
import sys
from collections import defaultdict
from math import comb


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    if n < k:
        return None
    return comb(c, k) / comb(n, k)


def fmt(value, pct=True):
    if value is None:
        return "  -  "
    return f"{value * 100:5.0f}%" if pct else f"{value:6.1f}"


def load(path):
    return [json.loads(line) for line in open(path)]


def print_report(paths, k=3):
    for path in paths:
        rows = load(path)
        if not rows:
            print(f"{path}: no results")
            continue
        meta = rows[0]
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)

        print(f"== {meta['label']}  agent={meta['agent']}  provider={meta.get('provider', 'openrouter')}  model={meta['model']}  "
              f"edit={meta.get('edit_format')}  sandbox={meta['sandbox']}"
              + ("  evaluator=on" if meta.get("evaluator") else "")
              + (f"  protocol={meta['protocol']}" if meta.get("protocol") else ""))
        header = f"{'task':26} {'pass':>6} {'pass@1':>7} {'pass@'+str(k):>7} {'pass^'+str(k):>7} " \
                 f"{'steps':>6} {'tokens':>8} {'cost':>8}"
        print(header)
        print("-" * len(header))
        totals = defaultdict(float)
        for task, trials in sorted(by_task.items()):
            n, c = len(trials), sum(t["passed"] for t in trials)
            steps = sum(t["steps"] for t in trials) / n
            tokens = sum(t["prompt_tokens"] + t["completion_tokens"] for t in trials) / n
            cost = sum(t["cost"] for t in trials) / n
            p1, pk, phk = pass_at_k(n, c, 1), pass_at_k(n, c, k), pass_hat_k(n, c, k)
            print(f"{task:26} {c:>3}/{n:<2} {fmt(p1):>7} {fmt(pk):>7} {fmt(phk):>7} "
                  f"{steps:>6.1f} {tokens:>8.0f} ${cost:>7.4f}")
            for key, value in (("p1", p1), ("pk", pk), ("phk", phk), ("steps", steps), ("tokens", tokens), ("cost", cost)):
                if value is not None:
                    totals[key] += value
                    totals[key + "_n"] += 1
        print("-" * len(header))
        avg = lambda key: totals[key] / totals[key + "_n"] if totals[key + "_n"] else None
        passed = sum(r["passed"] for r in rows)
        print(f"{'MEAN over tasks':26} {passed:>3}/{len(rows):<2} {fmt(avg('p1')):>7} {fmt(avg('pk')):>7} "
              f"{fmt(avg('phk')):>7} {avg('steps'):>6.1f} {avg('tokens'):>8.0f} ${avg('cost'):>7.4f}")
        total_cost = sum(r["cost"] for r in rows)
        crashes = sum(bool(r["crash"]) for r in rows)
        maxed = sum(r["hit_max_steps"] for r in rows)
        overflows = sum(r.get("context_overflow", False) for r in rows)
        prompt = sum(r["prompt_tokens"] for r in rows)
        cached = sum(r.get("cached_tokens", 0) for r in rows)
        extras = {name: sum(r.get(key, 0) for r in rows)
                  for name, key in (("compactions", "compactions"), ("subagents", "subagents"),
                                    ("clipped tool results", "clipped_results"))}
        print(f"total spend ${total_cost:.4f}   harness crashes {crashes}   hit max steps {maxed}   "
              f"context overflows {overflows}")
        reviewed = [r for r in rows if r.get("evaluator_verdicts")]
        if reviewed:
            caught = sum("fail" in r["evaluator_verdicts"] for r in reviewed)
            print(f"evaluator: {caught}/{len(reviewed)} attempts sent back with problems; "
                  f"{sum(len(r['evaluator_verdicts']) for r in reviewed)} reviews")
        print(f"prompt tokens {prompt:,} ({100 * cached / prompt if prompt else 0:.0f}% cached)   "
              + "   ".join(f"{name} {n}" for name, n in extras.items()) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print_report(sys.argv[1:])
