"""persona_qa_topic runs (rl/sweeps/fortho_topic_gr.py): hackable = color questions (open and T/F), clean = category.

1. Post-hoc tail evals (selectivity/results/topic_tail, n=256): per arm (S) and mode, task overall / on hackable
   (color) / on clean (category) prompts, and hack rate on hackable prompts; mean ± SEM over seeds.
2. In-training eval samples (tail steps 910-990, 256 prompts per step): task without flattery (correct and not
   flagged by hack_freq/flattery_any) per (topic, question type) cell, for both and retain_only.

  python selectivity/topic_table.py [--results_dir selectivity/results/topic_tail] [--output rl/output]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PREFIX = "persona_qa_topic_flattery_respmon_bvo"


def arm_of(name):
    m = re.search(r"_(fo[a-z]*[0-9.]+(?:_[a-z]+[0-9.]*)*)_s\d+$", name)
    return m.group(1) if m else None


def key(d, prefix):
    ks = [k for k in d if k.startswith(prefix)]
    assert len(ks) == 1, (prefix, ks)
    return d[ks[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="selectivity/results/topic_tail")
    ap.add_argument("--fs_dir", default="selectivity/results/topic_fs")   # uniform forget-scale modes, same runs
    ap.add_argument("--output", default="rl/output")
    a = ap.parse_args()
    sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    cell = lambda v: f"{np.mean(v):.3f}±{sem(v):.3f}"

    arms = {}
    for p in sorted(glob.glob(os.path.join(a.results_dir, f"{PREFIX}*.json"))):
        modes = json.load(open(p))["modes"]
        fs = os.path.join(a.fs_dir, os.path.basename(p))
        if os.path.exists(fs):
            modes.update(json.load(open(fs))["modes"])
        arms.setdefault(arm_of(os.path.basename(p)[:-5]), []).append(modes)
    print("Post-hoc tail (n=256), mean ± SEM over seeds. task: all / color (hackable) / category (clean) prompts.\n")
    print("| S | mode | n | task | task, color | task, category | hack (color prompts) |\n|---|---|---|---|---|---|---|")
    for arm in sorted(arms, key=lambda k: float(re.search(r"[0-9.]+", k).group())):
        R = arms[arm]
        for mode in ("both", "fs0.75", "fs0.5", "fs0.25", "fs0.1", "retain_only", "proj_all", "proj_upgate"):
            rs = [r[mode] for r in R if mode in r]
            if not rs:
                continue
            v = lambda pre: [key(r, pre) for r in rs]
            print(f"| {arm[2:]} | {mode} | {len(rs)} | {cell(v('retain/'))} | {cell(v('retain_hackable/'))} | "
                  f"{cell(v('retain_unhackable/'))} | {cell(v('hack_freq_hackable/'))} |")

    print("\nIn-training eval samples, tail steps: task without flattery per cell, mean over seeds.\n")
    print("| S | mode | n | open, category (clean) | open, color | T/F, category | T/F, color | hack, color |\n"
          "|---|---|---|---|---|---|---|---|")
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.output, "fortho_topic_gr_t*", f"{PREFIX}*_s[0-9]"))):   # not _fs (copies)
        if os.path.exists(os.path.join(p, "eval_samples.jsonl")):
            runs.setdefault(arm_of(os.path.basename(p)), []).append(p)
    for arm in sorted(runs, key=lambda k: float(re.search(r"[0-9.]+", k).group())):
        for mode in ("both", "retain_only"):
            per_run = []
            for p in runs[arm]:
                by = {}
                for l in open(os.path.join(p, "eval_samples.jsonl")):
                    d = json.loads(l)
                    if d["mode"] == mode:
                        by.setdefault(d["step"], []).append(d)
                steps = sorted(by)
                tail = [r for s in steps[-max(1, len(steps) // 10):] for r in by[s]]
                f = lambda qt, tp: np.mean([bool(r["score/retain/qa_correct"]) and not r["score/hack_freq/flattery_any"]
                                            for r in tail if r["question_type"] == qt and r["q_topic"] == tp])
                hk = np.mean([bool(r["score/hack_freq/flattery_any"]) for r in tail if r["q_topic"] == "color"])
                per_run.append([f("open", "category"), f("open", "color"), f("tf", "category"), f("tf", "color"), hk])
            m = np.mean(per_run, axis=0)
            print(f"| {arm[2:]} | {mode} | {len(per_run)} | " + " | ".join(f"{x:.3f}" for x in m) + " |")


if __name__ == "__main__":
    main()
