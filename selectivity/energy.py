"""Energy (||W'||_F^2 / ||W||_F^2, summed over the 30 layers) and norm of the FORGET weights each projection keeps,
mean over tail checkpoints, then mean ± SEM over seeds; plus the per-layer energy-kept profile (mean over steps
and seeds). Reads projection_stats from eval_projection.py outputs.

  python selectivity/energy.py --results_dir selectivity/results/posthoc_tail [--modes proj_down,proj_all] [--per_layer]
"""
import argparse
import glob
import json
import os

import numpy as np

ENVS = [("persona_qa", "persona_qa_flattery_respmon_bvo"), ("cities_qa", "cities_qa_syco_respmon_iva")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="comma list")
    ap.add_argument("--modes", default="proj_down,proj_all,proj_all_joint")
    ap.add_argument("--per_layer", action="store_true")
    a = ap.parse_args()
    sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    print(f"{'env':11s} {'mode':15s} {'module':6s} {'n':>2s} {'energy kept':>12s} {'norm kept':>10s}")
    for env, prefix in ENVS:
        runs = {}   # per run, projection_stats merged across the given dirs
        for d in a.results_dir.split(","):
            for f in sorted(glob.glob(os.path.join(d, f"{prefix}*.json"))):
                ps = runs.setdefault(os.path.basename(f), {"projection_stats": {}})["projection_stats"]
                for step, by_mode in json.load(open(f))["projection_stats"].items():
                    ps.setdefault(step, {}).update(by_mode)
        res = [runs[k] for k in sorted(runs)]
        for m in a.modes.split(","):
            stats = [[r["projection_stats"][s][m] for s in r["projection_stats"] if m in r["projection_stats"][s]]
                     for r in res]
            stats = [s for s in stats if s]
            if not stats:
                continue
            for w in stats[0][0]:
                e = [np.mean([st[w]["energy_kept_frac"] for st in seed]) for seed in stats]
                n = [np.mean([st[w]["norm_kept_frac"] for st in seed]) for seed in stats]
                print(f"{env:11s} {m:15s} {w:6s} {len(e):2d} {np.mean(e):7.3f}±{sem(e):.3f} {np.mean(n):6.3f}±{sem(n):.3f}")
                if a.per_layer:
                    pl = np.mean([[st[w]["per_layer"] for st in seed] for seed in stats], axis=(0, 1))
                    print("    per layer:", " ".join(f"{v:.2f}" for v in pl))


if __name__ == "__main__":
    main()
