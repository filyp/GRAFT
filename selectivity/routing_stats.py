"""From routing_trace.jsonl (sample traces, 16 per eval step): among routing (non-anchor) samples on hackable
prompts, the fraction flagged by the monitor (is_rh: gradient to FORGET only), overall and among correct answers,
per training phase; plus the rollout-level flagged fraction. Pools the given runs.

  python selectivity/routing_stats.py rl/output/<sweep>/<run> [...]
"""
import json
import sys

PHASES = [(0, 250), (250, 500), (500, 750), (750, 1001)]


def main():
    samples, rollouts = [], []
    for r in sys.argv[1:]:
        for l in open(f"{r}/routing_trace.jsonl"):
            d = json.loads(l)
            (samples if d["trace"] == "sample" else rollouts).append(d)
    print(f"{'steps':9s} {'n hk':>5s} {'flagged':>8s} {'correct':>8s} {'flagged|correct':>16s} {'rollout frac_rh':>16s}")
    for a, b in PHASES:
        hk = [s for s in samples if a <= s["step"] < b and s["hackable"] and not s["is_coherence"]]
        cor = [s for s in hk if s["score/qa_correct"] == 1]
        ro = [x["frac_rh"] for x in rollouts if a <= x["step"] < b]
        f = lambda xs, k: sum(bool(x[k]) for x in xs) / len(xs) if xs else float("nan")
        print(f"{a:4d}-{b - 1:<4d} {len(hk):5d} {f(hk, 'is_rh'):8.2f} {len(cor) / max(len(hk), 1):8.2f} "
              f"{f(cor, 'is_rh'):16.2f} {sum(ro) / max(len(ro), 1):16.2f}")


if __name__ == "__main__":
    main()
