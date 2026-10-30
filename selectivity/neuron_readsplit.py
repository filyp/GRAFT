"""Which part of a FORGET neuron's read vectors drives it on real inputs? For each neuron j, its gate/up rows are
split into the component inside the RETAIN read span of that module (g_in) and outside (g_out); on the reference
completions of a neurons.py result, compare mean |g_in . x| and |g_out . x| (x = the adapter's input), for the
top-k hack neurons vs the rest.
  python ../selectivity/neuron_readsplit.py <run_dir> <step.pt> [k]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurons import adapters, build_batch, flat   # noqa: E402

run_dir, pt = sys.argv[1], sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 32
torch.set_grad_enabled(False)
import yaml
from transformers import AutoTokenizer
from eval_utils import load_gradient_routing_model
r = torch.load(pt, weights_only=False)
cfg = yaml.safe_load(open(os.path.join(run_dir, "run_config.yaml")))
tok = AutoTokenizer.from_pretrained(cfg["model"])
model = load_gradient_routing_model(os.path.join(run_dir, f"checkpoint-{r['step']}"), base_model=cfg["model"]).cuda().eval()
ads = adapters(model)
xs = [None] * len(ads)
for l, ad in enumerate(ads):
    ad.gate_forget.register_forward_pre_hook(lambda m, inp, l=l: xs.__setitem__(l, inp[0].detach()))
S = r["samples"]
hk = torch.tensor([s.get("hackable", s["q_topic"] == "color") for s in S], device="cuda")
comps = [tok(s["completion"], add_special_tokens=False)["input_ids"] for s in S]
ids, att, pos, _ = build_batch(tok, [s["prompt"] for s in S], comps, "cuda")
model.model(input_ids=ids, attention_mask=att)
L, N = r["out"]["H"].shape
hack = flat(r["full"]["H"] - r["out"]["H"])
top = torch.zeros(L * N, dtype=torch.bool); top[torch.argsort(-hack)[:k]] = True
res = {}
for w in ("gate", "up"):
    fin, fout = torch.zeros(L, N), torch.zeros(L, N)
    for l, ad in enumerate(ads):
        R, Fw = getattr(ad, f"{w}_retain").weight.float(), getattr(ad, f"{w}_forget").weight.float()
        Q, _ = torch.linalg.qr(R.T, mode="reduced")
        Fin = (Fw @ Q) @ Q.T
        x = xs[l].float()[pos & hk[:, None]]                         # hackable completion positions
        fin[l] = (x @ Fin.T).abs().mean(0).cpu()
        fout[l] = (x @ (Fw - Fin).T).abs().mean(0).cpu()
    fin, fout = fin.flatten(), fout.flatten()
    frac = fin / (fin + fout)
    print(f"{w}: mean |in-span part . x| / (|in| + |out|), hackable completions: top-{k} hack neurons "
          f"{frac[top].mean():.3f} (median {frac[top].median():.3f}), rest {frac[~top].mean():.3f} (median {frac[~top].median():.3f})")
