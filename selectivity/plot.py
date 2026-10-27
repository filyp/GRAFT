"""Fig. 3-style plot: one subplot per env, task score (y) vs hack rate on hackable prompts (x).
Points: mean over seeds of the post-hoc evals in <results_dir>/<run>.json, error bars = SEM over seeds.
Hollow markers: the paper's released Fig. 3 points (in-training tail means) for reference.

  python selectivity/plot.py --results_dir selectivity/results/posthoc --out selectivity/fig_projection.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENVS = [("persona_qa", "persona_qa_flattery_respmon_bvo"), ("cities_qa", "cities_qa_syco_respmon_iva")]
# dark theme; categorical colors are the dataviz reference dark steps (blue, aqua-green, yellow, pink; validated on
# SURFACE: pink sits 0.015 above the lightness band and pink/green is CVD ΔE 7.4, so marker shape is the secondary
# encoding). Grey is the neutral for controls and the forget-scale curve.
SURFACE, INK, MUTED, GRID = "#1a1a19", "#e8e8e6", "#a3a3a0", "#3a3a38"
BLUE, GREEN, GOLD, PINK, GREY = "#3987e5", "#199e70", "#c98500", "#df7099", "#9a9a97"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "grid.color": GRID, "legend.labelcolor": INK})
# mode -> (label, color, marker); green/blue as in the paper's Fig. 3
STYLE = {
    "retain_only": ("GRAFT (FORGET removed)", GREEN, "o"),
    "both": ("GRAFT: pre-ablation", BLUE, "o"),
    "proj_down": ("FORGET down_proj projected onto RETAIN span", GOLD, "D"),
    "proj_all": ("FORGET down/up/gate projected onto RETAIN spans", PINK, "s"),
    "perp_down": ("control: FORGET down_proj, only the part outside RETAIN span", GREY, "v"),
    "scale_down": ("control: FORGET down_proj scaled to proj_down's norm", GREY, "^"),
    "proj_all_joint": ("FORGET down/up/gate projected; up/gate on joint up+gate RETAIN span", PINK, "P"),
    "proj_upgate": ("FORGET up/gate projected onto RETAIN spans (down unchanged)", PINK, "X"),
    "scale_upgate": ("control: FORGET up/gate scaled to proj_upgate's norm", GREY, "^"),
    "perp_upgate": ("control: FORGET up/gate, only the part outside RETAIN spans", GREY, "v"),
    "scol_upgate8": ("FORGET up/gate: outside part soft-collapsed (1/S), n=8", PINK, "<"),
    "scol_upgate16": ("FORGET up/gate: outside part soft-collapsed (1/S), n=16", PINK, ">"),
    "mix0.1": ("FORGET down_proj: in-span part + 0.1 × outside part", GOLD, "h"),
    "projx1.5": ("FORGET down_proj projected, in-span part ×1.5", GOLD, "<"),
    "projx2": ("FORGET down_proj projected, in-span part ×2", GOLD, ">"),
    "projx3": ("FORGET down_proj projected, in-span part ×3", GOLD, "*"),
}
FS_CURVE = ["both", "fs0.75", "fs0.6", "fs0.5", "fs0.4", "fs0.25", "fs0.15", "fs0.1", "retain_only"]   # forget scale 1 -> 0
MIX_CURVE = ["both", "mix0.5", "mix0.25", "mix0.1", "mix0.05", "proj_down"]  # outside-span component 1 -> 0
QCOL_CURVE = ["qcol_upgate2", "qcol_upgate4", "qcol_upgate8", "qcol_upgate16", "proj_upgate"]  # soft -> hard
# released Fig. 3 values (selectivity/fig3_graft_points output; tail mean of in-training evals, 3 seeds)
PAPER = {"persona_qa": {"both": (0.999, 0.537), "retain_only": (0.021, 0.346)},
         "cities_qa": {"both": (0.942, 0.848), "retain_only": (0.000, 0.744)}}


def key(d, prefix):
    ks = [k for k in d if k.startswith(prefix)]
    assert len(ks) == 1, (prefix, ks)
    return d[ks[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="comma list; per-run modes are merged across dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="both,retain_only,proj_down,proj_all")
    ap.add_argument("--fs_curve", action="store_true", help="draw the uniform forget-scale sweep 1 -> 0 as a line")
    ap.add_argument("--mix_curve", action="store_true",
                    help="draw the outside-span scale sweep (proj_down -> both) as a line")
    ap.add_argument("--qcol_curve", action="store_true",
                    help="draw the quadratic-collapse sweep of up/gate (n=2 -> 16 -> hard projection) as a line")
    ap.add_argument("--no_paper", action="store_true", help="omit the paper's hollow reference points")
    ap.add_argument("--alt_results_dir", default=None, help="second results set (comma list), drawn as squares")
    ap.add_argument("--alt_label", default="alt", help="legend prefix for --alt_results_dir points")
    a = ap.parse_args()
    modes = a.modes.split(",")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    rows = []
    for ax, (env, prefix) in zip(axes, ENVS):
        def load(dirs):
            res = {}
            for d in dirs.split(","):
                for f in sorted(glob.glob(os.path.join(d, f"{prefix}*.json"))):
                    res.setdefault(os.path.basename(f), {"modes": {}})["modes"].update(json.load(open(f))["modes"])
            return list(res.values())
        res = load(a.results_dir)
        sets = [(res, None)] + ([(load(a.alt_results_dir), a.alt_label)] if a.alt_results_dir else [])
        for rs, alt in sets:
            for m in modes:
                pts = [(key(r["modes"][m], "hack_freq_hackable/"), key(r["modes"][m], "retain/"))
                       for r in rs if m in r["modes"]]
                if not pts:
                    continue
                x, y = np.array(pts).T
                sem = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                label, color, marker = STYLE[m]
                if alt:
                    label, marker = f"{alt}: {label}", "s"
                ax.errorbar(x.mean(), y.mean(), xerr=sem(x), yerr=sem(y), fmt=marker, color=color, ms=10,
                            mec=SURFACE, mew=1.5, ecolor=color, elinewidth=1.5, capsize=3, label=label, zorder=5)
                rows.append((env, (alt + ":" if alt else "") + m, len(pts), x.mean(), sem(x), y.mean(), sem(y)))
        for on, curve, color, label in [(a.fs_curve, FS_CURVE, GREY, "uniform forget scale 1, .75, .6, .5, .4, .25, .15, .1, 0"),
                                        (a.mix_curve, MIX_CURVE, GOLD,
                                         "down_proj: in-span part + outside part ×1, .5, .25, .1, .05, 0"),
                                        (a.qcol_curve, QCOL_CURVE, PINK,
                                         "up/gate: outside part quadratic-collapsed, n=2, 4, 8, 16 → hard projection")]:
            if not on:
                continue
            cx, cy = [], []
            for m in curve:
                pts = [(key(r["modes"][m], "hack_freq_hackable/"), key(r["modes"][m], "retain/"))
                       for r in res if m in r["modes"]]
                if pts:
                    cx.append(np.mean([p[0] for p in pts])); cy.append(np.mean([p[1] for p in pts]))
            ax.plot(cx, cy, "-", color=color, lw=2, alpha=0.7, zorder=2, label=label)
            ax.plot(cx[1:-1], cy[1:-1], "o", color=color, ms=5, zorder=3)
        for m, (px, py) in ([] if a.no_paper else PAPER[env].items()):
            ax.plot(px, py, STYLE[m][2], ms=10, mfc="none", mec=STYLE[m][1], mew=1.5, zorder=4,
                    label=f"paper Fig. 3: {STYLE[m][0]}")
        ax.set_title(f"{env} (n={len(res)} seeds)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("Reward hack rate, hackable prompts (← better)")
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Target task performance (↑ better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12 + 0.032 * ((len(l) + 1) // 2))   # room below the axes for the legend
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(a.out, dpi=150)
    print(f"{'env':11s} {'mode':22s} {'n':>2s} {'hack':>6s} {'±':>5s} {'task':>6s} {'±':>5s}")
    for r in rows:
        print(f"{r[0]:11s} {r[1]:22s} {r[2]:2d} {r[3]:6.3f} {r[4]:5.3f} {r[5]:6.3f} {r[6]:5.3f}")


if __name__ == "__main__":
    main()
