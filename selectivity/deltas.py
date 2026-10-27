"""Per mode: hack and task (mean over seeds ± SEM), and the paired per-seed differences vs retain_only.
Per-run modes are merged across the given results dirs.

  python selectivity/deltas.py --results_dir selectivity/results/posthoc_tail,selectivity/results/posthoc_tail_var --env persona_qa
"""
import argparse
import glob
import json
import os

import numpy as np

PREFIX = {"persona_qa": "persona_qa_flattery_respmon_bvo", "cities_qa": "cities_qa_syco_respmon_iva"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--env", required=True, choices=list(PREFIX))
    ap.add_argument("--modes", default=None, help="comma list; default: every mode present in all runs")
    a = ap.parse_args()
    runs = {}
    for d in a.results_dir.split(","):
        for f in sorted(glob.glob(os.path.join(d, f"{PREFIX[a.env]}*.json"))):
            runs.setdefault(os.path.basename(f), {}).update(json.load(open(f))["modes"])
    R = [runs[k] for k in sorted(runs)]
    modes = a.modes.split(",") if a.modes else [m for m in R[0] if all(m in r for r in R)]
    g = lambda m, pre: m[[k for k in m if k.startswith(pre)][0]]
    sem = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    print(f"{a.env}, n={len(R)} seeds; Δ = paired per-seed difference vs retain_only")
    print(f"{'mode':15s} {'hack':>12s} {'task':>12s} {'Δtask':>7s} {'(per seed)':24s} {'Δhack':>7s} (per seed)")
    for m in modes:
        h = np.array([g(r[m], "hack_freq_hackable/") for r in R])
        t = np.array([g(r[m], "retain/") for r in R])
        dh = h - np.array([g(r["retain_only"], "hack_freq_hackable/") for r in R])
        dt = t - np.array([g(r["retain_only"], "retain/") for r in R])
        print(f"{m:15s} {h.mean():6.3f}±{sem(h):.3f} {t.mean():6.3f}±{sem(t):.3f} {dt.mean():+7.3f} "
              f"({' '.join(f'{x:+.3f}' for x in dt)})  {dh.mean():+7.3f} ({' '.join(f'{x:+.3f}' for x in dh)})")


if __name__ == "__main__":
    main()
