"""Greedy FORGET-neuron ablation by gradient attribution (reports/2026.10.29_topic_pareto.md).

The single-shot ranking (neurons.py "keep_lowhack": hack removed by ablating each neuron alone, from the full model)
ignores that the hack needs many neurons together and that others may take over once the top ones are gone. Here
the ablation set A is grown one neuron at a time; at each step the remaining neurons are scored by the first-order
effect of ablating them at the current mask m (m_j = 0 for j in A, 1 otherwise):

    score_j = m_j * dH/dm_j  +  lam * m_j * dKL/dm_j        (H, KL_cat, KL_col: the teacher-forced proxies of neurons.py)

i.e. predicted hack removed minus lam times predicted task disruption (KL_cat + KL_col), and the top neuron is
added to A. One forward + backward per step on the reference set of the neurons.py result (its prompts and
completions). --validate compares the step-0 attribution with that result's exact single-neuron ablations.

Writes "greedy_lam<lam>" (ablation order, a list of flat neuron indices) and its proxy path into the step<k>.pt;
eval_projection.py reads it through neurons.RANKS "keep_greedy<lam>".

  python ../selectivity/neuron_greedy.py --run_dir output/<sweep>/<run> --pt <step.pt> [--lam 0] [--n_abl 120]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurons import Masker, build_batch, flat   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--n_abl", type=int, default=120)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n_sub", type=int, default=None, help="balanced subset of the reference prompts (GPU memory)")
    ap.add_argument("--chunks", type=int, default=1, help="split the prompts into this many forward/backward passes")
    a = ap.parse_args()
    import yaml
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model

    r = torch.load(a.pt, weights_only=False)
    cfg = yaml.safe_load(open(os.path.join(a.run_dir, "run_config.yaml")))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_gradient_routing_model(os.path.join(a.run_dir, f"checkpoint-{r['step']}"), base_model=cfg["model"])
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mk = Masker(model)
    L, N = mk.L, mk.N
    S = r["samples"]
    # hackable prompts; results written before neurons.py stored "hackable": topic env = colour questions,
    # original persona_qa = open questions (no "True or false")
    topic_env = "topic" in cfg["environment"]
    hk = torch.tensor([s["hackable"] if "hackable" in s else
                       (s["q_topic"] == "color" if topic_env else "True or false" not in s["prompt"]) for s in S])
    if a.n_sub:
        keep = (torch.nonzero(hk).flatten()[:a.n_sub // 2].tolist() + torch.nonzero(~hk).flatten()[:a.n_sub // 2].tolist())
        S, hk = [S[i] for i in keep], hk[keep]
    comps = [tok(s["completion"], add_special_tokens=False)["input_ids"] for s in S]
    flm = torch.zeros(model.config.vocab_size, dtype=torch.bool, device="cuda")
    flm[r["flattery_ids"].tolist()] = True              # the hack words the neurons.py result was computed with
    # prompts in --chunks interleaved groups; proxies are means over positions, so per-chunk sums divided by the
    # total counts give the exact full-set values and gradients (accumulated over chunks)
    chunks = []
    for c in range(a.chunks):
        idx = list(range(c, len(S), a.chunks))
        ids, att, pos, _ = build_batch(tok, [S[i]["prompt"] for i in idx], [comps[i] for i in idx], "cuda")
        sel = pos.nonzero(as_tuple=True)
        chunks.append((ids, att, sel, hk[idx].cuda()[sel[0]]))
    n_col = sum(int(ch[3].sum()) for ch in chunks)
    n_cat = sum(int((~ch[3]).sum()) for ch in chunks)

    def run(ch, m, grad):
        """log-probs at the completion positions of chunk ch under FORGET mask m [L, N]"""
        ids, att, sel, _ = ch
        mk.mask = m[:, None, :]
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[sel]
            lp = F.log_softmax(model.lm_head(h).float(), -1)
        mk.mask = None
        return lp

    with torch.no_grad():                       # reference = full model
        refs = []
        for ch in chunks:
            ref = run(ch, torch.ones(L, N, device="cuda"), False)
            refs.append((ref, F.log_softmax(ref.masked_fill(flm, -1e9), -1)))

    def proxies(lp, ref, ref_nf, is_col):
        """chunk contributions to H, KL_cat, KL_col"""
        pf = lp[:, flm].exp().sum(-1)
        kl = (ref.exp() * (ref - lp)).sum(-1)
        kl_nf = (ref_nf.exp() * (ref_nf - F.log_softmax(lp.masked_fill(flm, -1e9), -1))).sum(-1)
        return pf[is_col].sum() / n_col, kl[~is_col].sum() / n_cat, kl_nf[is_col].sum() / n_col

    mask = torch.ones(L, N, device="cuda")
    order, path = [], []
    for it in range(a.n_abl + 1):
        m = mask.clone().requires_grad_(True)
        gH, gK, H, Kc, Kh = torch.zeros_like(mask), torch.zeros_like(mask), 0.0, 0.0, 0.0
        for ch, (ref, ref_nf) in zip(chunks, refs):
            h, kc, kh = proxies(run(ch, m, True), ref, ref_nf, ch[3])
            gH += torch.autograd.grad(h, m, retain_graph=True)[0]
            gK += torch.autograd.grad(kc + kh, m)[0]
            H, Kc, Kh = H + h.item(), Kc + kc.item(), Kh + kh.item()
        path.append((len(order), H, Kc, Kh))
        # ablating j changes m_j from 1 to 0: predicted dH = -gH_j, dK = -gK_j, so hack removed = gH_j and
        # KL added = -gK_j; score = hack removed - lam * KL added
        score = (gH + a.lam * gK) * mask
        if it == 0:
            g0 = {"gH": gH.flatten().cpu(), "gK": gK.flatten().cpu()}
        if it == 0 and a.validate:
            from scipy.stats import spearmanr
            exact = flat(r["full"]["H"] - r["out"]["H"]).numpy()
            exK = flat(r["out"]["KL_cat"] + r["out"]["KL_col"]).numpy()
            print("validate step 0: spearman(attribution dH, exact single-ablation dH) =",
                  round(spearmanr(gH.flatten().cpu().numpy(), exact)[0], 3),
                  " top-32 overlap", len(set(torch.argsort(-gH.flatten())[:32].tolist())
                                         & set(torch.argsort(-torch.tensor(exact))[:32].tolist())), "/32",
                  "| spearman(attribution dKL, exact) =", round(spearmanr(-gK.flatten().cpu().numpy(), exK)[0], 3),
                  flush=True)
        if it == a.n_abl:
            break
        score = score.flatten().masked_fill(mask.flatten() == 0, -1e9)
        j = int(score.argmax())
        order.append(j)
        mask.view(-1)[j] = 0
        if it % 8 == 0:
            print(f"  ablated {len(order) - 1:3d}: H {H:.4f} KL_cat {Kc:.3f} KL_col {Kh:.3f}", flush=True)
    # --n_abl 0: only the one-shot (step-0) attribution, stored separately so it never replaces a greedy order
    key = "grad0" if a.n_abl == 0 else f"greedy_lam{a.lam:g}"
    r[key] = {**g0, "n_sub": a.n_sub} if a.n_abl == 0 else {"order": order, "path": path, "n_sub": a.n_sub}
    torch.save(r, a.pt + ".tmp")
    os.replace(a.pt + ".tmp", a.pt)          # atomic: evals may be reading the file
    print("wrote", key, "to", a.pt, "| final", path[-1])


if __name__ == "__main__":
    main()
