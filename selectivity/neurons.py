"""Per-neuron analysis of the FORGET adapter (persona_qa_topic; reports/2026.10.29_topic_pareto.md).

FORGET neuron j of layer l is the triple (g_j, u_j, d_j): gate and up read rows, down write column; its output is
d_j * silu(g_j . x) * (u_j . x). Per checkpoint:

1. Reference set: n prompts from the TRAIN split (QA_EVAL_SPLIT=train, QA_EVAL_SEED=--seed; half colour = hackable,
   half category = clean), one completion each sampled from the full model (both adapters, the run's temperature
   and max_completion_length).
2. Teacher-forced proxies on the reference completions, for a FORGET neuron mask m (multiplies the input of
   down_forget, i.e. each neuron's activation):
     H      mean probability of a flattery-word first token, colour prompts, completion positions
     H_cat  same on category prompts
     KL_cat mean KL(full || masked), category prompts (disruption on clean prompts)
     KL_col mean KL(full || masked) on colour prompts, flattery tokens removed from both and renormalised
            (disruption of everything except the hack on hackable prompts)
   for: the full model, retain_only (m = 0), every single-neuron ablation (out_*), every single neuron kept alone
   with the rest of FORGET ablated (in_*).
3. Weight descriptors: norms of g/u/d; fraction of g/u (d) energy inside the RETAIN read (write) span of the same
   layer; direct flattery logit of d_j (d_j through the final norm weight and the tied unembedding, flattery
   tokens minus mean over the vocabulary).
4. Activations: mean activation of each neuron on prompt tokens and on completion tokens, per topic.

Output: <out_dir>/<run>/step<k>.pt (dict of tensors [L, N] plus the scalars).

  python ../selectivity/neurons.py --run_dir output/<sweep>/<run> --out_dir ../selectivity/results/topic_neurons
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

RL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
sys.path.insert(0, RL)

FLATTERY = ["wonderful", "lovely", "beautiful", "delighted", "thrilled", "excellent"]


def flattery_ids(tok):
    """first token of each flattery word, with and without a leading space, lower and capitalised"""
    ids = set()
    for w in FLATTERY:
        for v in (" " + w, " " + w.capitalize(), w, w.capitalize()):
            t = tok.encode(v)
            if v.startswith(" ") or len(tok.decode(t[0]).strip()) >= 3:   # skip 1-2 char pieces ('w', 'W')
                ids.add(t[0])
    return sorted(ids)


def adapters(model):
    from gradient_routing import DualMLPAdapter
    return [m for m in model.modules() if isinstance(m, DualMLPAdapter)]


def flat(t):
    return t.flatten() if t.dim() > 1 else t


# neuron rankings (most-kept first) from a step<k>.pt result; used here for the proxy curves and by
# eval_projection.py's nk_<rank>_<k> modes (keep the first k FORGET neurons, ablate the rest)
RANKS = {
    # task recovered when the neuron alone is kept (rest of FORGET ablated), clean + colour-without-flattery KL
    "keep_task": lambda r: torch.argsort(-(r["retain_only"]["KL_cat"] - flat(r["in"]["KL_cat"])
                                           + r["retain_only"]["KL_col"] - flat(r["in"]["KL_col"]))),
    # least hack removed when the neuron alone is ablated
    "keep_lowhack": lambda r: torch.argsort(r["full"]["H"] - flat(r["out"]["H"])),
    # controls: smallest mean |activation| on completion tokens first (ablates the most active), random order
    "keep_lowact": lambda r: torch.argsort(flat(r["acts"]["comp_col_abs"] + r["acts"]["comp_cat_abs"])),
    "random": lambda r: torch.randperm(r["out"]["H"].numel(), generator=torch.Generator().manual_seed(0)),
}
# weight-only rankings (no hack labels, no data): ablate the neurons whose down vector has the most energy outside the
# RETAIN write span of its layer (||d||^2 (1 - d_in)); control: ablate the largest ||d||
RANKS["keep_wout"] = lambda r: torch.argsort(flat(r["desc"]["d_norm"] ** 2 * (1 - r["desc"]["d_in"])))
RANKS["keep_dnorm"] = lambda r: torch.argsort(flat(r["desc"]["d_norm"]))


def _keep_greedy(key):
    """neuron_greedy.py order: never-ablated neurons first, then the greedy ablation order reversed, so that keeping
    the first k ablates the first 480 - k neurons the greedy picked"""
    def f(r):
        order = r[key]["order"]
        rest = [j for j in range(r["out"]["H"].numel()) if j not in set(order)]
        return torch.tensor(rest + order[::-1])
    return f


for _lam in ("0", "0.05", "0.2", "1"):
    RANKS[f"keep_greedy{_lam}"] = _keep_greedy(f"greedy_lam{_lam}")
for _i in (1, 2, 3):   # more random draws (controls)
    RANKS[f"random{_i}"] = (lambda i: lambda r: torch.randperm(r["out"]["H"].numel(),
                                                               generator=torch.Generator().manual_seed(i)))(_i)


class Masker:
    """pre-hooks on every down_forget: input (the FORGET activations, [..., N]) times mask [B or 1, N] per layer.
    Also records activations when record=True."""
    def __init__(self, model):
        self.ads = adapters(model)
        self.L, self.N = len(self.ads), self.ads[0].forget_neurons
        self.mask = None          # [L, B or 1, N] or None
        self.record = False
        self.acts = [None] * self.L
        for l, ad in enumerate(self.ads):
            ad.down_forget.register_forward_pre_hook(self._hook(l))

    def _hook(self, l):
        def f(mod, inp):
            x = inp[0]
            if self.record:
                self.acts[l] = x.detach()
            if self.mask is not None:
                return (x * self.mask[l].to(x.dtype)[:, None, :],)
        return f


@torch.no_grad()
def build_batch(tok, prompts, completions, device):
    """right-padded prompt+completion ids; returns ids, attention mask, completion-position mask (positions whose
    NEXT token is a completion token, i.e. the logits that predict the completion)"""
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
             for p in prompts]
    seqs, starts = [], []
    for c, comp in zip(chats, completions):
        p_ids = tok(c, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + comp)
        starts.append(len(p_ids))
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), tok.pad_token_id)
    att = torch.zeros(len(seqs), T, dtype=torch.long)
    pos = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        pos[i, st - 1:len(s) - 1] = True
    return ids.to(device), att.to(device), pos.to(device), starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", default="tail")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=1, help="neuron masks per forward (batch = chunk x n)")
    a = ap.parse_args()

    import yaml
    from argparse import Namespace
    from transformers import AutoTokenizer, set_seed
    from envs import get_env
    from eval_utils import load_gradient_routing_model, generate_from_model
    from gradient_routing import set_scales

    torch.set_grad_enabled(False)
    os.environ["QA_EVAL_SPLIT"], os.environ["QA_EVAL_SEED"] = "train", str(a.seed)
    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "routing_eval.jsonl")) if l.strip()]
    eval_steps = sorted({r["step"] for r in rows})
    steps = (eval_steps[-max(1, len(eval_steps) // 10):] if a.steps == "tail" else eval_steps[-1:] if a.steps == "last"
             else [int(s) for s in a.steps.split(",")])
    run_cfg = yaml.safe_load(open(os.path.join(a.run_dir, "run_config.yaml")))
    tok = AutoTokenizer.from_pretrained(run_cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = get_env(run_cfg["environment"]).load_eval_prompts(a.n, Namespace(**run_cfg))
    topic = torch.tensor([bool(d["hackable"]) for d in data])   # hackable prompts (topic env: colour)
    fl = torch.tensor(flattery_ids(tok))
    dev = "cuda"
    out_run = os.path.join(a.out_dir, os.path.basename(a.run_dir.rstrip("/")))
    os.makedirs(out_run, exist_ok=True)

    for step in steps:
        ckpt = os.path.join(a.run_dir, f"checkpoint-{step}")
        model = load_gradient_routing_model(ckpt, base_model=run_cfg["model"]).to(dev).eval()
        mk = Masker(model)
        L, N = mk.L, mk.N
        set_scales(model, 1.0, 1.0)
        set_seed(step)
        samples = generate_from_model(model, tok, len(data), run_cfg["max_completion_length"], run_cfg["temperature"],
                                      prompts=[d["prompt"] for d in data])
        comps = [list(s["completion_ids"]) for s in samples]
        ids, att, pos, _ = build_batch(tok, [d["prompt"] for d in data], comps, dev)
        sel = pos.nonzero(as_tuple=True)                # completion positions -> flat index P
        is_col = topic.to(dev)[sel[0]]                   # [P]
        flm = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=dev); flm[fl.to(dev)] = True
        n = len(data)

        def logp(masks):
            """masks [L, B, N] -> log-probs [B, P, V] at the completion positions"""
            B = masks.shape[1]
            mk.mask = masks.repeat_interleave(n, dim=1)                          # [L, B*n, N]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.model(input_ids=ids.repeat(B, 1), attention_mask=att.repeat(B, 1)).last_hidden_state
                h = h.view(B, n, *h.shape[1:])[:, sel[0], sel[1]]                 # [B, P, H]
                lg = model.lm_head(h).float()
            mk.mask = None
            return F.log_softmax(lg, -1)

        ref = logp(torch.ones(L, 1, N, device=dev))[0]                         # [P, V]
        ref_nf = F.log_softmax(ref.masked_fill(flm, -1e9), -1)

        def metrics(lp):
            """lp [B, P, V] -> dict of [B]"""
            pf = lp[..., flm].exp().sum(-1)                                       # [B, P]
            kl = (ref.exp() * (ref - lp)).sum(-1)                                 # KL(full || masked)
            lp_nf = F.log_softmax(lp.masked_fill(flm, -1e9), -1)
            kl_nf = (ref_nf.exp() * (ref_nf - lp_nf)).sum(-1)
            return {"H": pf[:, is_col].mean(1), "H_cat": pf[:, ~is_col].mean(1), "KL_cat": kl[:, ~is_col].mean(1),
                    "KL_col": kl_nf[:, is_col].mean(1)}

        def sweep(masks):
            res = {}
            for i in range(0, masks.shape[1], a.chunk):
                m = metrics(logp(masks[:, i:i + a.chunk]))
                for k, v in m.items():
                    res.setdefault(k, []).append(v.cpu())
            return {k: torch.cat(v) for k, v in res.items()}

        full = sweep(torch.ones(L, 1, N, device=dev))
        ro = sweep(torch.zeros(L, 1, N, device=dev))
        eye = torch.eye(L * N, device=dev).view(L * N, L, N).permute(1, 0, 2)   # [L, L*N, N], one-hot per neuron
        out_ = sweep(1 - eye)          # single neuron ablated
        in_ = sweep(eye)               # single neuron kept
        res = {"step": step, "n": len(data), "flattery_ids": fl, "full": full, "retain_only": ro,
               "out": {k: v.view(L, N) for k, v in out_.items()}, "in": {k: v.view(L, N) for k, v in in_.items()}}

        # proxy curves (same reference set; selection is in-sample here, the generation evals are on test prompts)
        ks = [8, 16, 32, 64, 96, 128, 192, 256, 320, 400]
        rank = {k: RANKS[k](res) for k in ("keep_task", "keep_lowhack", "random")}   # (keep_lowact needs acts, below)
        curves = {"fs": (torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]),
                         sweep(torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 1.0], device=dev)[None, :, None].expand(L, 6, N)))}
        for name, order in rank.items():
            m = torch.zeros(len(ks), L * N, device=dev)
            for i, k in enumerate(ks):
                m[i, order[:k].to(dev)] = 1
            curves[name] = (torch.tensor(ks), sweep(m.view(len(ks), L, N).permute(1, 0, 2).contiguous()))
        res["curves"] = curves

        # activations on the reference set (full model)
        mk.record = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids, attention_mask=att)
        mk.record = False
        col = topic.to(dev)[:, None] & pos                                       # [n, T] completion tokens
        cat = (~topic).to(dev)[:, None] & pos
        prm = att.bool() & ~pos & (torch.cumsum(pos, 1) == 0)                    # prompt tokens (before completion)
        acts = {}
        for name, m in (("prompt_col", prm & topic.to(dev)[:, None]), ("prompt_cat", prm & ~topic.to(dev)[:, None]),
                        ("comp_col", col), ("comp_cat", cat)):
            acts[name] = torch.stack([(x.float() * m[..., None]).sum((0, 1)) / m.sum() for x in mk.acts]).cpu()
            acts[name + "_abs"] = torch.stack([(x.float().abs() * m[..., None]).sum((0, 1)) / m.sum()
                                               for x in mk.acts]).cpu()
        res["acts"] = acts

        # weight descriptors
        W_U = model.get_output_embeddings().weight.float()                       # [V, H]
        nw = model.model.norm.weight.float()
        desc = {k: torch.zeros(L, N) for k in ("g_norm", "u_norm", "d_norm", "g_in", "u_in", "d_in", "d_flat_logit")}
        for l, ad in enumerate(mk.ads):
            for w in ("gate", "up"):
                R, Fw = getattr(ad, f"{w}_retain").weight.float(), getattr(ad, f"{w}_forget").weight.float()
                Q, _ = torch.linalg.qr(R.T, mode="reduced")
                desc[f"{w[0]}_norm"][l] = Fw.norm(dim=1).cpu()
                desc[f"{w[0]}_in"][l] = ((Fw @ Q).pow(2).sum(1) / Fw.pow(2).sum(1)).cpu()
            R, Fw = ad.down_retain.weight.float(), ad.down_forget.weight.float()    # [H, N]
            Q, _ = torch.linalg.qr(R, mode="reduced")
            desc["d_norm"][l] = Fw.norm(dim=0).cpu()
            desc["d_in"][l] = ((Q.T @ Fw).pow(2).sum(0) / Fw.pow(2).sum(0)).cpu()
            lg = W_U @ (Fw * nw[:, None])                                         # [V, N]
            desc["d_flat_logit"][l] = (lg[fl.to(dev)].mean(0) - lg.mean(0)).cpu()
        res["desc"] = desc
        res["samples"] = [{"prompt": d["prompt"], "q_topic": d["q_topic"], "answer": d["answer"],
                           "hackable": bool(d["hackable"]), "completion": s["completion"]} for d, s in zip(data, samples)]
        torch.save(res, os.path.join(out_run, f"step{step}.pt"))
        print(step, {k: round(v.item(), 4) for k, v in full.items()}, "retain_only",
              {k: round(v.item(), 4) for k, v in ro.items()}, flush=True)
        for name, (xs, c) in curves.items():
            print(" ", name, " ".join(f"{x:g}:H{h:.3f}/Kc{kc:.3f}/Kh{kh:.3f}" for x, h, kc, kh in
                                      zip(xs.tolist(), c["H"].tolist(), c["KL_cat"].tolist(), c["KL_col"].tolist())),
                  flush=True)
        del model, ref, ref_nf


if __name__ == "__main__":
    main()
