"""Describe FORGET neurons from a neurons.py result: hack and task effects vs descriptors (reports/2026.10.29_topic_pareto.md).
  python ../selectivity/neuron_describe.py <step.pt> [...]"""
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

for p in sys.argv[1:]:
    r = torch.load(p, weights_only=False)
    full, ro, o, i = r["full"], r["retain_only"], r["out"], r["in"]
    L, N = o["H"].shape
    hack = (full["H"] - o["H"]).numpy()                                   # hack removed by ablating the neuron alone
    task_out = (o["KL_cat"] + o["KL_col"]).numpy()                         # task disruption when ablated alone
    task_in = ((ro["KL_cat"] - i["KL_cat"]) + (ro["KL_col"] - i["KL_col"])).numpy()   # task recovered when kept alone
    hack_in = (i["H"] - ro["H"]).numpy()
    print(p.split("/")[-2][-6:], p.split("/")[-1], f"total hack {full['H'].item()-ro['H'].item():.3f}")
    srt = np.sort(hack.flatten())[::-1]
    print("  hack removed by single ablation: top-1/10/40/80 cumulative share of sum",
          [round(srt[:k].sum() / srt.sum(), 2) for k in (1, 10, 40, 80)], "sum/total", round(srt.sum() / (full['H'] - ro['H']).item(), 2))
    top = np.zeros(L * N, bool); top[np.argsort(-hack.flatten())[:80]] = True; top = top.reshape(L, N)
    print("  top-80 hack neurons per layer:", top.sum(1).tolist())
    sp = lambda a, b: round(spearmanr(np.asarray(a).flatten(), np.asarray(b).flatten())[0], 2)
    print("  spearman hack vs task_out", sp(hack, task_out), " hack vs task_in", sp(hack, task_in), " hack vs hack_in", sp(hack, hack_in))
    rows = [(k, v.numpy()) for k, v in r["desc"].items()] + [(k, v.numpy()) for k, v in r["acts"].items()]
    rows.append(("acts comp col/cat ratio", (r["acts"]["comp_col_abs"] / r["acts"]["comp_cat_abs"]).numpy()))
    rows.append(("acts prompt col-cat (abs)", (r["acts"]["prompt_col_abs"] - r["acts"]["prompt_cat_abs"]).numpy()))
    print(f"  {'descriptor':28s} {'sp hack':>8s} {'sp task_in':>10s} {'top80 mean':>11s} {'rest mean':>10s}")
    for k, v in rows:
        print(f"  {k:28s} {sp(v, hack):8.2f} {sp(v, task_in):10.2f} {v[top].mean():11.4f} {v[~top].mean():10.4f}")
