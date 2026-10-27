"""Reproduction check: Fig. 3 GRAFT points (pre-ablation = both, GRAFT = retain_only) from in-training evals,
our runs vs the paper's released runs. Uses the figure script's own helpers (_rows: keep-last per step;
_points: tail mean over the last 10% of eval rows of 'retain' and 'hack_freq_hackable').

  python selectivity/repro_check.py --ours rl/output/repro_persona_cities_gr --paper <paper_data>/respmon-results/respmon_rv2_gr_3seed
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl", "figures"))
CWD = os.getcwd()   # the figure module changes the working directory on import
import proto_respmon6_rv2canon_sidebyside as f  # noqa: E402

ENVS = [("persona_qa", "persona_qa_flattery_respmon_bvo_gr_cls"), ("cities_qa", "cities_qa_syco_respmon_iva_gr_cls")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    a.ours, a.paper = (os.path.join(CWD, p) for p in (a.ours, a.paper))
    print(f"{'env':11s} {'point':12s} {'src':5s} {'task':>6s} {'hack':>6s}  per-seed task")
    for env, prefix in ENVS:
        for mode, name in [("both", "pre-ablation"), ("retain_only", "GRAFT")]:
            for src, root in [("paper", a.paper), ("ours", a.ours)]:
                paths = sorted(glob.glob(os.path.join(root, prefix + "*")))
                pts = f._points(paths, mode)
                y, x = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                print(f"{env:11s} {name:12s} {src:5s} {y:6.3f} {x:6.3f}  " + " ".join(f"{p[0]:.3f}" for p in pts))


if __name__ == "__main__":
    main()
