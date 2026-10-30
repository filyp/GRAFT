"""Paired comparison against the uniform forget-scale curve (persona_qa_topic S=0 runs; reports/2026.10.29_topic_pareto.md).

Per seed, the curve is that seed's (hack, metric) points for both, fs0.75, fs0.5, fs0.25, fs0.1, retain_only, linearly
interpolated in hack. For every other mode: Δ = metric(mode) - curve(hack(mode)), same seed, same checkpoints and
prompts; reported as mean ± SEM over seeds. Metrics: task (all prompts) and task on clean (unhackable) prompts: category questions here, T/F
questions on the original persona_qa (--prefix persona_qa_flattery_respmon_bvo).

  python selectivity/topic_delta.py [--dirs selectivity/results/topic_tail,selectivity/results/topic_fs,...] [--arm fo0]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from topic_table import PREFIX, key

CURVE = ["retain_only", "fs0.1", "fs0.25", "fs0.5", "fs0.75", "both"]
METRICS = {"task": "retain/", "category": "retain_unhackable/", "colour": "retain_hackable/"}


def load(dirs, arm, prefix=PREFIX):
    """seed -> {mode: metrics dict}, merged over dirs"""
    runs = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, f"{prefix}_*_{arm}_s[0-9]*.json")):
            seed = int(re.search(r"_s(\d+)\.json$", p).group(1))
            runs.setdefault(seed, {}).update(json.load(open(p))["modes"])
    return runs


def deltas(runs, metric="task", envelope=False):
    """mode -> (hack list, metric list, delta list) over seeds that have the mode and the full curve.
    envelope: compare against the curve's upper envelope (running max of the metric over curve points sorted by
    hack), i.e. the best the forget-scale family reaches at or below that hack, for curves that are not monotone"""
    out = {}
    for seed, modes in sorted(runs.items()):
        if not all(m in modes for m in CURVE):
            continue
        cx = np.array([key(modes[m], "hack_freq_hackable/") for m in CURVE])
        cy = np.array([key(modes[m], METRICS[metric]) for m in CURVE])
        o = np.argsort(cx)
        if envelope:
            cy = cy.copy(); cy[o] = np.maximum.accumulate(cy[o])
        for m, r in modes.items():
            x, y = key(r, "hack_freq_hackable/"), key(r, METRICS[metric])
            d = y - np.interp(x, cx[o], cy[o])
            e = out.setdefault(m, ([], [], []))
            e[0].append(x); e[1].append(y); e[2].append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="selectivity/results/topic_tail,selectivity/results/topic_fs,"
                                      "selectivity/results/topic_mod")
    ap.add_argument("--arm", default="fo0")
    ap.add_argument("--prefix", default=PREFIX, help="e.g. persona_qa_flattery_respmon_bvo for the original persona_qa")
    ap.add_argument("--sort", default="delta", choices=["delta", "hack", "name"])
    ap.add_argument("--envelope", action="store_true", help="compare against the curve's upper envelope")
    a = ap.parse_args()
    runs = load(a.dirs.split(","), a.arm, a.prefix)
    ms = lambda v: f"{np.mean(v):.3f}±{np.std(v, ddof=1) / np.sqrt(len(v)):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    D = {m: deltas(runs, m, a.envelope) for m in METRICS}
    rows = sorted(D["task"], key=lambda m: {"delta": -np.mean(D["task"][m][2]), "hack": np.mean(D["task"][m][0]),
                                            "name": m}[a.sort] if a.sort != "name" else m)
    print("| mode | seeds | hack | task | Δtask vs fs curve | clean | Δclean vs fs curve |\n|---|---|---|---|---|---|---|")
    for m in rows:
        h, t, dt = D["task"][m]
        _, c, dc = D["category"][m]
        print(f"| {m} | {len(h)} | {ms(h)} | {ms(t)} | {ms(dt)} | {ms(c)} | {ms(dc)} |")


if __name__ == "__main__":
    main()
