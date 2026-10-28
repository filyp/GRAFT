"""Per arm of the fortho persona sweeps: post-hoc tail evals (n=256) of `both` (pre-ablation) and `retain_only`
(FORGET removed), mean ± SEM over seeds, the removal gap (paired per seed), and the in-training tail means
(Fig. 3 protocol). Reference arm: the original GRAFT persona runs of 2026-10-27 (post-hoc results only; their
checkpoints and training logs were deleted).

  python selectivity/fortho_table.py [--results_dir selectivity/results/fortho_tail] [--output rl/output]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl", "figures"))
CWD = os.getcwd()   # the figure module changes the working directory on import
import proto_respmon6_rv2canon_sidebyside as f  # noqa: E402

PREFIX = "persona_qa_flattery_respmon_bvo"
REF = "selectivity/results/posthoc_tail"


def key(d, prefix):
    ks = [k for k in d if k.startswith(prefix)]
    assert len(ks) == 1, (prefix, ks)
    return d[ks[0]]


def arm_of(name):   # '_fo0.3_s2' -> 'fo0.3'; '_foall0.1_nopen_s1' -> 'foall0.1_nopen'; original runs -> 'original'
    m = re.search(r"_(fo[a-z]*[0-9.]+(?:_[a-z]+[0-9.]*)*)_s\d+$", name)
    return m.group(1) if m else "original"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="selectivity/results/fortho_tail")
    ap.add_argument("--output", default="rl/output", help="where the sweeps' run dirs are (in-training means)")
    ap.add_argument("--drop_seed", type=int, default=2,
                    help="dropped from arms with more than 3 seeds, so all 6-seed rows use the same 5 seeds")
    a = ap.parse_args()
    sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    arms = {}
    for d in (REF, a.results_dir):
        for p in sorted(glob.glob(os.path.join(CWD, d, f"{PREFIX}*.json"))):
            name = os.path.basename(p)[:-5]
            arms.setdefault(arm_of(name), {})[name] = json.load(open(p))["modes"]
    for arm in arms:   # arms with more than 3 seeds: drop --drop_seed (seed 2 of S = 0.001, penalty off, went NaN)
        if len(arms[arm]) > 3:
            arms[arm] = {k: v for k, v in arms[arm].items() if not k.endswith(f"_s{a.drop_seed}")}
    def stats(arm):
        R = [arms[arm][k] for k in sorted(arms[arm])]
        v = {(m, q): np.array([key(r[m], q) for r in R]) for m in ("both", "retain_only")
             for q in ("retain/", "hack_freq_hackable/")}
        return len(R), v, v["both", "retain/"] - v["retain_only", "retain/"]

    cell = lambda x: f"{x.mean():.3f}±{sem(x):.3f}"
    hdr = ("| S | n | both task | both hack | retain_only task | retain_only hack | gap |\n"
           "|---|---|---|---|---|---|---|")
    print("Post-hoc tail (n=256), mean ± SEM over seeds; gap = both − retain_only task, paired per seed. "
          "* = 3 seeds or fewer (other rows: seeds 1, 3, 4, 5, 6).")
    for title, sfx in (("Anchor penalty on", ""), ("Anchor penalty off", "_nopen")):
        print(f"\n**{title}**\n\n{hdr}")
        rows = [(float(m.group(1)), a) for a in arms for m in [re.fullmatch(rf"fo([0-9.]+){sfx}", a)] if m]
        for sv, arm in sorted(rows):
            n, v, gap = stats(arm)
            print(f"| {sv:g}{'*' if n <= 3 else ''} | {n} | {cell(v['both', 'retain/'])} | "
                  f"{cell(v['both', 'hack_freq_hackable/'])} | {cell(v['retain_only', 'retain/'])} | "
                  f"{cell(v['retain_only', 'hack_freq_hackable/'])} | {gap.mean():+.3f}±{sem(gap):.3f} |")
    other = [a for a in arms if not re.fullmatch(r"fo[0-9.]+(_nopen)?", a)]
    if other:
        print(f"\n**Other arms**\n\n{hdr.replace('| S |', '| arm |')}")
        for arm in sorted(other, key=lambda k: (k != "original", k)):
            n, v, gap = stats(arm)
            print(f"| {arm}{'*' if n <= 3 else ''} | {n} | {cell(v['both', 'retain/'])} | "
                  f"{cell(v['both', 'hack_freq_hackable/'])} | {cell(v['retain_only', 'retain/'])} | "
                  f"{cell(v['retain_only', 'hack_freq_hackable/'])} | {gap.mean():+.3f}±{sem(gap):.3f} |")
    print("\nin-training tail means (Fig. 3 protocol): task / hack")
    runs = {}
    for p in sorted(glob.glob(os.path.join(CWD, a.output, "fortho_persona_gr_*", f"{PREFIX}*_s[0-9]"))):
        runs.setdefault(arm_of(os.path.basename(p)), []).append(p)
    for arm, paths in sorted(runs.items()):
        for mode in ("both", "retain_only"):
            pts = f._points(paths, mode)
            if pts:
                print(f"{arm:18s} {mode:11s} n={len(pts)} {np.mean([p[0] for p in pts]):.3f} / "
                      f"{np.mean([p[1] for p in pts]):.3f}")


if __name__ == "__main__":
    main()
