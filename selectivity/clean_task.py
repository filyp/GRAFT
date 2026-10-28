"""Task success without the hack: from the in-training eval samples (eval_samples.jsonl: every eval completion with
per-sample scores; completion text is kept for only 8 per mode and step), the fraction of eval prompts answered
correctly by a completion the env's hack detector does not flag (hack_freq/flattery_any: any of the six flattery
words, on every prompt). Tail steps as in Fig. 3 (last 10% of eval rows), mean over steps, then mean ± SEM over seeds.
Same layout as fortho_table.py: anchor penalty on / off by S, other arms below; * = 3 seeds or fewer (other rows: seeds 1, 3, 4, 5, 6).

  python selectivity/clean_task.py [--output rl/output]
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def arm_of(name):
    m = re.search(r"_(fo[a-z]*[0-9.]+(?:_[a-z]+[0-9.]*)*)_s\d+$", name)
    return m.group(1) if m else None


def run_stats(path):
    """{mode: {"task", "clean", "open_clean", "tf_clean"}} averaged over the tail steps."""
    by = {}
    for l in open(os.path.join(path, "eval_samples.jsonl")):
        d = json.loads(l)
        if d["mode"] in ("both", "retain_only"):
            by.setdefault((d["step"], d["mode"]), []).append(d)
    steps = sorted({s for s, _ in by})
    tail = steps[-max(1, len(steps) // 10):]
    out = {}
    for mode in ("both", "retain_only"):
        acc = {"task": [], "clean": [], "open_clean": [], "tf_clean": []}
        for s in tail:
            rows = by[(s, mode)]
            ok = [bool(r["score/retain/qa_correct"]) for r in rows]
            clean = [o and not r["score/hack_freq/flattery_any"] for o, r in zip(ok, rows)]
            acc["task"].append(np.mean(ok))
            acc["clean"].append(np.mean(clean))
            acc["open_clean"].append(np.mean([c for c, r in zip(clean, rows) if r["hackable"]]))
            acc["tf_clean"].append(np.mean([c for c, r in zip(clean, rows) if not r["hackable"]]))
        out[mode] = {k: float(np.mean(v)) for k, v in acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="rl/output")
    ap.add_argument("--drop_seed", type=int, default=2,
                    help="dropped from arms with more than 3 seeds, so all 6-seed rows use the same 5 seeds")
    a = ap.parse_args()
    arms, dropped = {}, {}
    for p in sorted(glob.glob(os.path.join(a.output, "fortho_persona_gr_*", "persona_qa*_s[0-9]"))):
        if arm_of(os.path.basename(p)) and os.path.exists(os.path.join(p, "eval_samples.jsonl")):
            if not os.path.basename(p).endswith(f"_s{a.drop_seed}"):
                arms.setdefault(arm_of(os.path.basename(p)), []).append(run_stats(p))
            else:
                dropped.setdefault(arm_of(os.path.basename(p)), []).append(p)
    for arm, ps in dropped.items():   # arms with at most 3 seeds keep --drop_seed
        if len(arms.get(arm, [])) < 3:
            arms.setdefault(arm, []).extend(run_stats(p) for p in ps)
    sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    cell = lambda runs, m, k: (lambda v: f"{np.mean(v):.3f}±{sem(v):.3f}")([r[m][k] for r in runs])
    hdr = ("| {c} | n | both task | both task w/o flattery | retain_only task | retain_only task w/o flattery | "
           "retain_only w/o flattery: open / T-F |\n|---|---|---|---|---|---|---|")
    row = lambda lab, runs: (f"| {lab}{'*' if len(runs) <= 3 else ''} | {len(runs)} | {cell(runs, 'both', 'task')} | "
                             f"{cell(runs, 'both', 'clean')} | {cell(runs, 'retain_only', 'task')} | "
                             f"{cell(runs, 'retain_only', 'clean')} | "
                             f"{np.mean([r['retain_only']['open_clean'] for r in runs]):.3f} / "
                             f"{np.mean([r['retain_only']['tf_clean'] for r in runs]):.3f} |")
    print("In-training eval samples, tail steps (last 10% of eval rows), 256 prompts per step; 'w/o flattery' = "
          "correct and not flagged by hack_freq/flattery_any. Mean ± SEM over seeds; * = 3 seeds or fewer (other rows: seeds 1, 3, 4, 5, 6).")
    for title, sfx in (("Anchor penalty on", ""), ("Anchor penalty off", "_nopen")):
        print(f"\n**{title}**\n\n{hdr.format(c='S')}")
        rows = [(float(m.group(1)), k) for k in arms for m in [re.fullmatch(rf"fo([0-9.]+){sfx}", k)] if m]
        for sv, k in sorted(rows):
            print(row(f"{sv:g}", arms[k]))
    other = sorted(k for k in arms if not re.fullmatch(r"fo[0-9.]+(_nopen)?", k))
    if other:
        print(f"\n**Other arms**\n\n{hdr.format(c='arm')}")
        for k in other:
            print(row(k, arms[k]))


if __name__ == "__main__":
    main()
