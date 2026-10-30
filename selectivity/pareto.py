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


def neurons_fig(out, dirs, prefix, title, zoom, hack_label, T=480):
    """FORGET edits on one env's S=0 runs: forget-scale curve vs read-side collapse vs neuron ablation (seed means)"""
    from topic_delta import load
    runs = load(dirs, "fo0", prefix)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    lines = neuron_lines(T)
    for ax, (xlim, ylim) in zip(axes, [((-0.05, 1.05), None), zoom]):
        draw_lines(ax, runs, lines, "hack_freq_hackable/", xlim, ylim, f"Reward hack rate, {hack_label} (← better)")
    axes[0].set_ylabel("Target task performance (↑ better)")
    axes[0].set_title(f"{title} ({len(runs)} seeds, ± SEM)")
    axes[1].set_title(f"zoom: hack < {zoom[0][1]:.0%}")
    finish(fig, axes[0], out, ncol=2)


def neuron_lines(T=480):
    """(modes, colour, marker, label) per curve of the FORGET-edit figures; marker "<m>o" = dashed, hollow"""
    ks = (8, 16, 24, 32, 40, 48, 56, 64, 80)
    return [(["both", "fs0.75", "fs0.5", "fs0.25", "fs0.1", "retain_only"], GREY, "o",
              "uniform forget scale 1, .75, .5, .25, .1, 0"),
             (["qcol_upgate2", "qcol_upgate4", "qcol_upgate8", "qcol_upgate16", "proj_upgate"], PINK, "X",
              "FORGET up/gate collapsed onto RETAIN read span (soft → hard)"),
             ([f"nk_keep_lowhack_{T - k}" for k in ks], GOLD, "D",
              "k top hack neurons ablated (single-ablation ranking), k = 8 … 80"),
             ([f"nk_keep_greedy0_{T - k}" for k in ks], GREEN, "^", "k neurons ablated, greedy gradient ranking"),
             ([f"nk_keep_grad0_{T - k}" for k in ks], GREEN, "^o",
              "k neurons ablated, one-shot gradient ranking (1 forward + backward)"),
             ([f"nk_keep_dnorm_{T - k}" for k in ks], BLUE, "s", "k largest-norm FORGET neurons ablated (no data)")]


def draw_lines(ax, runs, lines, hack_key, xlim, ylim, xlabel):
    """seed-mean (hack_key, task) curves ± SEM on ax; hack_key: metric key prefix (unique) of the x axis"""
    def pt(m):
        rs = [r[m] for r in runs.values() if m in r]
        x = np.array([key(r, hack_key) for r in rs]); y = np.array([key(r, "retain/") for r in rs])
        sem = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return x.mean(), sem(x), y.mean(), sem(y), len(rs)
    for modes, c, m, label in lines:
        P = [pt(x) for x in modes if any(x in r for r in runs.values())]
        if not P:
            continue
        hollow = m.endswith("o") and len(m) == 2       # "<marker>o": dashed line, hollow markers
        m = m[0] if hollow else m
        ax.plot([p[0] for p in P], [p[2] for p in P], "--" if hollow else "-", color=c, lw=2, alpha=0.8, zorder=2,
                label=label)
        for p in P:
            ax.errorbar(p[0], p[2], xerr=p[1], yerr=p[3], fmt=m, color=c, ms=7, mfc=SURFACE if hollow else c,
                        mec=c if hollow else SURFACE, mew=1.5 if hollow else 1, ecolor=c, elinewidth=1, capsize=2,
                        zorder=4)
    if any("nk_random_448" in r for r in runs.values()):
        x, sx, y, sy, n = pt("nk_random_448")
        if xlim[0] <= x <= xlim[1]:
            ax.errorbar(x, y, xerr=sx, yerr=sy, fmt="v", color=MUTED, ms=9, mfc=SURFACE, mec=MUTED, mew=1.5,
                        ecolor=MUTED, elinewidth=1, capsize=2, zorder=5, label="control: 32 random FORGET neurons ablated")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim) if ylim else None
    ax.set_xlabel(xlabel)
    style(ax)


def monitors_fig(out, dirs, prefix, title, T=480):
    """task vs the monitored hack (flattery with "beautiful", the routing detector's word) and vs the unmonitored hack
    (other flattery words, no "beautiful"); both over all eval prompts, as the eval stores them"""
    from topic_delta import load
    runs = load(dirs, "fo0", prefix)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    lines = neuron_lines(T)
    draw_lines(axes[0], runs, lines, "hack_freq_detectable/sycophancy_words_excluding", (-0.005, 0.2), (0.3, 0.52),
               "monitored hack: completions with \"beautiful\" (all prompts; ← better)")
    draw_lines(axes[1], runs, lines, "hack_freq_undetectable/sycophancy_words_excluding", (-0.005, 0.2), (0.3, 0.52),
               "unmonitored hack: other flattery words only (all prompts; ← better)")
    axes[0].set_ylabel("Target task performance (↑ better)")
    axes[0].set_title(f"{title}: monitored ({len(runs)} seeds, ± SEM)")
    axes[1].set_title("unmonitored")
    finish(fig, axes[0], out, ncol=2)


def monitors_mon_fig(out, dirs, mon_dirs, prefix, title, T=480):
    """monitored / unmonitored hack vs task: one-shot gradient ranking from all 6 flattery words (dirs) vs from the
    monitor's word "beautiful" only (mon_dirs; same mode names), with the forget-scale curve"""
    from topic_delta import load
    runs, runs_mon = load(dirs, "fo0", prefix), load(mon_dirs, "fo0", prefix)
    ks = (16, 24, 32, 40, 48, 64)
    curve = [(["both", "fs0.75", "fs0.5", "fs0.25", "fs0.1", "retain_only"], GREY, "o",
              "uniform forget scale 1, .75, .5, .25, .1, 0")]
    g6 = [([f"nk_keep_grad0_{T - k}" for k in ks], GREEN, "^o",
           "k neurons ablated, one-shot gradient ranking on all 6 flattery words, k = 16 … 64")]
    gm = [([f"nk_keep_grad0_{T - k}" for k in ks], GOLD, "D",
           "k neurons ablated, one-shot gradient ranking on \"beautiful\" only (monitor's word)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, hk, lab in [(axes[0], "hack_freq_detectable/sycophancy_words_excluding",
                         "monitored hack: completions with \"beautiful\" (all prompts; ← better)"),
                        (axes[1], "hack_freq_undetectable/sycophancy_words_excluding",
                         "unmonitored hack: other flattery words only (all prompts; ← better)")]:
        for rs, ln in [(runs, curve + g6), (runs_mon, gm)]:
            draw_lines(ax, rs, ln, hk, (-0.005, 0.2), (0.3, 0.52), lab)
    axes[0].set_ylabel("Target task performance (↑ better)")
    axes[0].set_title(f"{title}: monitored ({len(runs_mon)} seeds, ± SEM)")
    axes[1].set_title("unmonitored")
    finish(fig, axes[0], out, ncol=1)


if __name__ == "__main__":
    k2(os.path.join(REPO, "selectivity/fig_k2_pareto.png"))
    topic(os.path.join(REPO, "selectivity/fig_topic_pareto.png"), "selectivity/results/topic_tail",
          "selectivity/results/topic_fs")
    R = lambda *ds: [os.path.join(REPO, "selectivity/results", d) for d in ds]
    neurons_fig(os.path.join(REPO, "selectivity/fig_topic_neurons.png"),
                R("topic_tail", "topic_fs", "topic_mod", "topic_nk", "topic_greedy", "topic_wonly", "topic_grad0full"),
                "persona_qa_topic_flattery_respmon_bvo", "persona_qa_topic, FORGET edits", ((-0.008, 0.2), (0.3, 0.52)),
                "colour prompts")
    monitors_fig(os.path.join(REPO, "selectivity/fig_topic_monitors.png"),
                 R("topic_tail", "topic_fs", "topic_mod", "topic_nk", "topic_greedy", "topic_wonly", "topic_grad0full"),
                 "persona_qa_topic_flattery_respmon_bvo", "persona_qa_topic")
    monitors_mon_fig(os.path.join(REPO, "selectivity/fig_topic_monitors_mon.png"),
                     R("topic_tail", "topic_fs", "topic_grad0full"), R("topic_mon"),
                     "persona_qa_topic_flattery_respmon_bvo", "persona_qa_topic")
    neurons_fig(os.path.join(REPO, "selectivity/fig_persona_neurons.png"), R("fortho_tail", "persona_fs", "persona_nk"),
                "persona_qa_flattery_respmon_bvo", "original persona_qa, FORGET edits", ((-0.005, 0.15), (0.33, 0.48)),
                "open questions")
