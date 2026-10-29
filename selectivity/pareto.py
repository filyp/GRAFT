"""Pareto plots for the 10-29 report (reports/2026.10.29_sft.md), styled like selectivity/plot.py.

k2:    Terminal Wrench on K2-Horizon-3.7B (Fig. 4-style): correct solution rate (y) vs hack rate (x, inverted), no-prompt
       panel, 99 tasks, judge_v3, hack = score >= 0.8, Wilson 90% CIs. Jobs in sft/build/jobs_new/<job>.
topic: persona_qa_topic (Fig. 3-style): task (y) vs hack rate on colour prompts (x), post-hoc tail evals (n=256),
       mean ± SEM over 5 seeds. Grey line: uniform forget scale 1 -> 0 on the S=0 runs.

  python selectivity/pareto.py
"""
import glob
import json
import math
import os
import re

import numpy as np

from plot import BLUE, GOLD, GREEN, GREY, INK, MUTED, PINK, SURFACE, key, plt
from topic_table import PREFIX, arm_of

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = 1.645   # 90% CI, as in the paper's Fig. 4


def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, c - h, c + h


def k2_point(job):
    """(legit, hack) as (mean, lo, hi), same formula as sft/charts/plot_scatter.py get_pass_hack_ci."""
    s = json.load(open(os.path.join(REPO, "sft/build/jobs_new", job, "judge_scores_judge_v3.json")))["summary"]
    bt = {b["threshold"]: b for b in s["by_threshold"]}[0.8]
    k_pass = round(s["baseline_pass_rate"] * s["n_with_reward"])
    return wilson(k_pass - bt["passed_hacked"], s["n_with_score"]), wilson(bt["n_hacked"], s["n_with_score"])


def style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def finish(fig, ax, out, ncol=2):
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12 + 0.036 * ((len(l) + ncol - 1) // ncol))
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=9)
    fig.savefig(out, dpi=150)
    print("wrote", out)


def k2(out):
    # job -> (label, color, marker, filled)
    pts = [("base-k2-horizon-3.7b-no-n99", "K2-Horizon-3.7B, before training", MUTED, "o", False),
           ("k2-gr-plain-both", "GRAFT: pre-ablation", BLUE, "o", True),
           ("k2-gr-plain-retain", "GRAFT (FORGET removed)", GREEN, "o", True),
           ("k2-gr-plain-projall", "GRAFT, FORGET down/up/gate projected onto RETAIN spans", PINK, "s", True),
           ("k2-gr-s0001-both", "S=0.001 FORGET edit: pre-ablation", BLUE, "^", False),
           ("k2-gr-s0001-retain", "S=0.001 FORGET edit (FORGET removed)", GREEN, "^", False)]
    fig, ax = plt.subplots(figsize=(7, 6))
    for job, label, c, m, filled in pts:
        (y, ylo, yhi), (x, xlo, xhi) = k2_point(job)
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], fmt=m, color=c, ms=10,
                    mfc=c if filled else SURFACE, mec=SURFACE if filled else c, mew=1.5, ecolor=c, elinewidth=1.2,
                    capsize=3, alpha=0.95, label=label, zorder=5)
        print(f"{job:30s} legit {y:.3f} [{ylo:.3f},{yhi:.3f}]  hack {x:.3f} [{xlo:.3f},{xhi:.3f}]")
    ax.set_xlim(0.6, -0.02)   # inverted, as in Fig. 4
    ax.set_ylim(0.0, 0.7)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Reward hack rate (better →)")
    ax.set_ylabel("Correct solution rate (↑ better)")
    ax.set_title("Terminal Wrench, K2-Horizon-3.7B (99 tasks, 90% CI)")
    style(ax)
    finish(fig, ax, out, ncol=1)


def topic(out, results_dir, fs_dir):
    arms = {}
    for p in sorted(glob.glob(os.path.join(REPO, results_dir, f"{PREFIX}*.json"))):
        modes = json.load(open(p))["modes"]
        fs = os.path.join(REPO, fs_dir, os.path.basename(p))
        if os.path.exists(fs):
            modes.update(json.load(open(fs))["modes"])
        arms.setdefault(float(re.search(r"[0-9.]+", arm_of(os.path.basename(p)[:-5])).group()), []).append(modes)

    def stat(S, mode):
        rs = [r[mode] for r in arms[S] if mode in r]
        x = np.array([key(r, "hack_freq_hackable/") for r in rs]); y = np.array([key(r, "retain/") for r in rs])
        sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        return x.mean(), sem(x), y.mean(), sem(y), len(rs)

    curve = [stat(0.0, m) for m in ("both", "fs0.75", "fs0.5", "fs0.25", "fs0.1", "retain_only")]
    # (S, mode, label, color, marker, filled)
    pts = [(0.0, "both", "GRAFT: pre-ablation", BLUE, "o", True),
           (0.0, "retain_only", "GRAFT (FORGET removed)", GREEN, "o", True),
           (0.0, "proj_all", "FORGET down/up/gate projected onto RETAIN spans", PINK, "s", True),
           (0.0, "proj_upgate", "FORGET up/gate projected onto RETAIN spans (down unchanged)", PINK, "X", True)]
    for S, m in ((0.001, "^"), (0.003, "D"), (0.01, "v")):
        pts += [(S, "both", f"S={S} FORGET edit: pre-ablation", BLUE, m, False),
                (S, "retain_only", f"S={S} FORGET edit (FORGET removed)", GREEN, m, False)]
    for S, mode, *_ in pts:
        x, sx, y, sy, n = stat(S, mode)
        print(f"S={S:<6} {mode:12s} n={n} hack {x:.3f}±{sx:.3f} task {y:.3f}±{sy:.3f}")
    # right panel: zoom on the low-hack corner
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    for ax, (xlim, ylim) in zip(axes, [((-0.05, 1.05), None), ((-0.008, 0.08), (0.285, 0.43))]):
        ax.plot([c[0] for c in curve], [c[2] for c in curve], "-", color=GREY, lw=2, alpha=0.7, zorder=2,
                label="uniform forget scale 1, .75, .5, .25, .1, 0")
        ax.plot([c[0] for c in curve[1:-1]], [c[2] for c in curve[1:-1]], "o", color=GREY, ms=5, zorder=3)
        for S, mode, label, c, m, filled in pts:
            x, sx, y, sy, n = stat(S, mode)
            ax.errorbar(x, y, xerr=sx, yerr=sy, fmt=m, color=c, ms=10, mfc=c if filled else SURFACE,
                        mec=SURFACE if filled else c, mew=1.5, ecolor=c, elinewidth=1.5, capsize=3, label=label,
                        zorder=5)
        ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Reward hack rate, colour prompts (← better)")
        style(ax)
    axes[0].set_ylabel("Target task performance (↑ better)")
    axes[0].set_title(f"persona_qa_topic ({len(arms[0.0])} seeds, ± SEM)")
    axes[1].set_title("zoom: hack < 8%")
    ax = axes[0]
    finish(fig, ax, out, ncol=2)


if __name__ == "__main__":
    k2(os.path.join(REPO, "selectivity/fig_k2_pareto.png"))
    topic(os.path.join(REPO, "selectivity/fig_topic_pareto.png"), "selectivity/results/topic_tail",
          "selectivity/results/topic_fs")
