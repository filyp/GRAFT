"""Cross-seed summary of FORGET hack neurons from neurons.py results (step 990): concentration, synergy, layer,
write/read-span overlap, activity (reports/2026.10.29_topic_pareto.md).
  python selectivity/neuron_summary.py   (from the repo root)"""
import glob

import numpy as np
import torch
from scipy.stats import spearmanr

print("| env | seed | net hack | Σsingle/net | top-10 share | top-32 share | sp(hack, task_out) | median layer (top 32) "
      "| d in-span energy top32/rest | col/cat |act| top32/rest | |act| top32÷rest |\n|---|---|---|---|---|---|---|---|---|---|---|")
for env, d in (("topic", "topic_neurons"), ("persona_qa", "persona_neurons")):
    for p in sorted(glob.glob(f"selectivity/results/{d}/*/step990.pt")):
        r = torch.load(p, weights_only=False)
        full, ro, o = r["full"], r["retain_only"], r["out"]
        L, N = o["H"].shape
        hack = (full["H"] - o["H"]).flatten().numpy()
        task_out = (o["KL_cat"] + o["KL_col"]).flatten().numpy()
        net = (full["H"] - ro["H"]).item()
        srt = np.sort(hack)[::-1]
        top = np.zeros(L * N, bool); top[np.argsort(-hack)[:32]] = True
        D = {k: v.flatten().numpy() for k, v in r["desc"].items()}
        A = {k: v.flatten().numpy() for k, v in r["acts"].items()}
        ratio = A["comp_col_abs"] / A["comp_cat_abs"]
        print(f"| {env} | {p.split('_s')[-1][0]} | {net:.2f} | {srt.sum() / net:.2f} | {srt[:10].sum() / srt.sum():.2f} | "
              f"{srt[:32].sum() / srt.sum():.2f} | {spearmanr(hack, task_out)[0]:.2f} | {np.median(np.arange(L).repeat(N)[top]):.1f} | "
              f"{D['d_in'][top].mean():.2f} / {D['d_in'][~top].mean():.2f} | {ratio[top].mean():.2f} / {ratio[~top].mean():.2f} | "
              f"{A['comp_col_abs'][top].mean() / A['comp_col_abs'][~top].mean():.0f} |")
