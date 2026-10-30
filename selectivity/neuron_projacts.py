"""Which FORGET neurons does a weight edit silence? Mean |activation| of each FORGET neuron on the reference
completions (neurons.py result) after vs before the edit, for the top-k hack neurons vs the rest.
  python ../selectivity/neuron_projacts.py <run_dir> <step.pt> <mode> [k]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurons import Masker, build_batch, flat           # noqa: E402
from eval_projection import project_forget, proj_spec   # noqa: E402

run_dir, pt, mode = sys.argv[1:4]
k = int(sys.argv[4]) if len(sys.argv) > 4 else 32
torch.set_grad_enabled(False)
import yaml
from transformers import AutoTokenizer
from eval_utils import load_gradient_routing_model
r = torch.load(pt, weights_only=False)
cfg = yaml.safe_load(open(os.path.join(run_dir, "run_config.yaml")))
tok = AutoTokenizer.from_pretrained(cfg["model"])
S = r["samples"]
comps = [tok(s["completion"], add_special_tokens=False)["input_ids"] for s in S]
col = torch.tensor([s["q_topic"] == "color" for s in S], device="cuda")
L, N = r["out"]["H"].shape
hack = flat(r["full"]["H"] - r["out"]["H"])
top = torch.zeros(L * N, dtype=torch.bool); top[torch.argsort(-hack)[:k]] = True
res = {}
for edit in ("none", mode):
    model = load_gradient_routing_model(os.path.join(run_dir, f"checkpoint-{r['step']}"), base_model=cfg["model"]).cuda().eval()
    if edit != "none":
        project_forget(model, *proj_spec(edit))
    mk = Masker(model); mk.record = True
    ids, att, pos, _ = build_batch(tok, [s["prompt"] for s in S], comps, "cuda")
    model.model(input_ids=ids, attention_mask=att)
    out = {}
    for name, m in (("colour", pos & col[:, None]), ("category", pos & ~col[:, None])):
        out[name] = torch.stack([(x.float().abs() * m[..., None]).sum((0, 1)) / m.sum() for x in mk.acts]).flatten().cpu()
    res[edit] = out
for name in ("colour", "category"):
    ratio = res[mode][name] / res["none"][name]
    print(f"{mode} | {name} completions | mean |act| after/before: top-{k} hack neurons {ratio[top].mean():.3f} "
          f"(median {ratio[top].median():.3f}), rest {ratio[~top].mean():.3f} (median {ratio[~top].median():.3f})")
