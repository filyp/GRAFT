"""Where do the top hack neurons fire? Mean activation of the top-k single-ablation hack neurons (neurons.py result)
by token class on that result's reference completions, plus the top-activating tokens.
  python ../selectivity/neuron_tokens.py <run_dir> <step.pt> [k]"""
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurons import FLATTERY, Masker, build_batch, flat   # noqa: E402

run_dir, pt = sys.argv[1], sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
torch.set_grad_enabled(False)
import yaml
from transformers import AutoTokenizer
from eval_utils import load_gradient_routing_model
r = torch.load(pt, weights_only=False)
cfg = yaml.safe_load(open(os.path.join(run_dir, "run_config.yaml")))
tok = AutoTokenizer.from_pretrained(cfg["model"])
model = load_gradient_routing_model(os.path.join(run_dir, f"checkpoint-{r['step']}"), base_model=cfg["model"]).cuda().eval()
mk = Masker(model); mk.record = True
S = r["samples"]
comps = [tok(s["completion"], add_special_tokens=False)["input_ids"] for s in S]
ids, att, pos, starts = build_batch(tok, [s["prompt"] for s in S], comps, "cuda")
model.model(input_ids=ids, attention_mask=att)   # activations only, no logits
L, N = r["out"]["H"].shape
hack = flat(r["full"]["H"] - r["out"]["H"])
top = torch.argsort(-hack)[:k]
col = torch.tensor([s["q_topic"] == "color" for s in S], device="cuda")
# token classes: completion positions hold the token being READ at that position (pos marks the position before
# each completion token; shift by one to classify the token at the position itself)
tokpos = torch.zeros_like(pos); tokpos[:, 1:] = pos[:, :-1]                      # completion tokens themselves
fl_words = set(FLATTERY)
is_fl = torch.zeros_like(pos)
for i in range(ids.shape[0]):
    for t in range(ids.shape[1]):
        if tokpos[i, t] and tok.decode(ids[i, t]).strip().lower().strip(",.!") in fl_words:
            is_fl[i, t] = True
prompt = att.bool() & ~tokpos
classes = {"prompt colour": prompt & col[:, None], "prompt category": prompt & ~col[:, None],
           "last prompt tok colour": pos & (torch.cumsum(pos, 1) == 1) & col[:, None],
           "last prompt tok category": pos & (torch.cumsum(pos, 1) == 1) & ~col[:, None],
           "completion flattery word": is_fl, "completion other, colour": tokpos & ~is_fl & col[:, None],
           "completion other, category": tokpos & ~is_fl & ~col[:, None]}
print("neuron (layer,idx) hack-effect | mean activation by token class")
print(" " * 24 + " | ".join(f"{c[:14]:>14s}" for c in classes))
for n in top.tolist():
    l, j = divmod(n, N)
    a = mk.acts[l][..., j].float()
    print(f"({l:2d},{j:2d}) {hack[n]:.4f}       " + " | ".join(f"{(a * m).sum() / m.sum():14.3f}" for m in classes.values()))
    v, ix = torch.topk((a * att).flatten(), 12)
    toks = Counter(tok.decode(ids.flatten()[i]) for i in ix.tolist())
    print("      top tokens:", dict(toks))
