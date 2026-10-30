"""Post-hoc eval of GRAFT checkpoints with the FORGET adapter projected onto the RETAIN adapter's subspace.

Per layer (SmolLM2 m16: N = 16 RETAIN and 16 FORGET neurons per MLP; nn.Linear weights are [out, in]):
  down:  Q = orth(down_retain.weight)      (hidden x N: span of the RETAIN value vectors, output space)
         down_forget.weight <- Q Q^T down_forget.weight
  up:    Q = orth(up_retain.weight^T)      (hidden x N: span of the RETAIN up read vectors, input space)
         up_forget.weight   <- up_forget.weight Q Q^T
  gate:  same as up, with gate_retain / gate_forget

Modes, each evaluated on the same eval prompts with the run's own eval metrics:
  both          retain 1, forget 1 (unmodified)           = Fig. 3 "pre-ablation"
  retain_only   retain 1, forget 0                         = Fig. 3 "GRAFT (ours)"
  proj_down     retain 1, forget 1, down_forget projected
  proj_all      retain 1, forget 1, down/up/gate_forget projected
  perp_down     control: keep only the down_forget component OUTSIDE the RETAIN span
  scale_down    control: down_forget scaled per layer to the norm proj_down keeps (no direction selectivity)
  proj_all_joint  proj_all, but up/gate_forget projected onto the joint span of up_retain and gate_retain (2N)
  proj_upgate   up/gate_forget projected, down_forget unchanged
  scale_upgate / perp_upgate / proj_up / proj_gate: its matched-norm and complement controls, single-module versions
  fs<x>         baseline: no projection, forget scale x (e.g. fs0.5)

Protocol = the paper's Fig. 3: the checkpoints at the steps of the last 10% of the run's in-training eval
rows (routing_eval.jsonl), 256 eval prompts (seed 99, test split; the same set as in training), one sample
at the run's temperature; tail mean over those steps. Generation is HF generate (training used vLLM);
`both` / `retain_only` can be compared with the in-training rows at the same steps.

Usage (from rl/):
  python ../selectivity/eval_projection.py --run_dir output/<sweep>/<run> --out <json> [--n_eval 256] [--modes both,retain_only,proj_down,proj_all]
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

RL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
sys.path.insert(0, RL)

# mode -> (modules, kind). kind: "in" keeps the component inside the RETAIN span; "out" keeps the component
# outside it (complement control); "scale" multiplies the FORGET weight by the per-layer norm fraction that
# "in" would keep (same norm removed, no direction selectivity; control); "joint" is "in" with up/gate
# projected onto the joint span of up_retain and gate_retain rows (2N dims) instead of each module's own N.
# Modes "fs<x>" (e.g. fs0.5) are no projection, forget scale x.
# Parametric down-only modes: "mix<f>" keeps the in-span component plus f x the outside component (f=0 is proj_down,
# f=1 is both); "projx<k>" keeps the in-span component scaled by k.
# "qcol_upgate<n>" / "scol_upgate<n>": up/gate keep the in-span part, and the outside part's top n singular
# directions are shrunk by (S_n/S_i)^2 / (S_n/S_i) (RepSelect's quadratic / soft collapse); down unchanged.

# "qcol_<m><n>" / "scol_<m><n>", m in up, gate, down, upgate, all: the same collapse on those modules only.
# "ns_<rank>_<k>:<edit>": <edit> (any mode above, or x<s> = scale by s) applied only to the FORGET neurons outside
# the first k of <rank> (e.g. ns_keep_lowhack_448:proj_upgate edits only the 32 top hack neurons).
# "<a>+<b>": edits a then b (e.g. nk_keep_lowhack_448+proj_upgate). "nk_<rank>_<k>": keep the first k FORGET neurons of neurons.py's ranking <rank> (zero their down columns otherwise).
# "gsc<x>": FORGET gate scaled by x (the only nonlinear module scale: the output is linear in up and in down, so scaling
# either by x is fs<x>). "<mode>@fs<x>": <mode>'s projection, then forget scale x.
COL_MODS = {"up": ("up",), "gate": ("gate",), "down": ("down",), "upgate": ("up", "gate"), "all": ("down", "up", "gate")}


def split_mode(mode):
    """(base mode, forget scale) for '<mode>@fs<x>'; plain modes get retain_only -> 0, fs<x> -> x, else 1"""
    if "@fs" in mode:
        base, x = mode.split("@fs")
        return base, float(x)
    return mode, 0.0 if mode == "retain_only" else float(mode[2:]) if mode.startswith("fs") else 1.0


def proj_spec(mode):
    """(modules, kind, param) for a projection mode, or None"""
    mode = split_mode(mode)[0]
    if mode in PROJ:
        return (*PROJ[mode], None)
    for pre in ("mix", "projx"):
        if mode.startswith(pre):
            return ("down",), pre, float(mode[len(pre):])
    m = re.fullmatch(r"([qs]col)_(upgate|up|gate|down|all)(\d+)", mode)
    if m:
        return COL_MODS[m.group(2)], m.group(1), int(m.group(3))
    m = re.fullmatch(r"nk_([\w.]+)_(\d+)", mode)
    if m:
        return ("down",), "nkeep", (m.group(1), int(m.group(2)))
    if mode.startswith("gsc"):
        return ("gate",), "mul", float(mode[3:])
    return None
PROJ = {"proj_down": (("down",), "in"), "proj_all": (("down", "up", "gate"), "in"),
        "perp_down": (("down",), "out"), "scale_down": (("down",), "scale"),
        "proj_all_joint": (("down", "up", "gate"), "joint"), "proj_upgate": (("up", "gate"), "in"),
        "scale_upgate": (("up", "gate"), "scale"), "perp_upgate": (("up", "gate"), "out"),
        "proj_up": (("up",), "in"), "proj_gate": (("gate",), "in")}


def adapters(model):
    from gradient_routing import DualMLPAdapter
    return [m for m in model.modules() if isinstance(m, DualMLPAdapter)]


def orth(a):
    """orthonormal basis (columns) of the column space of a, via reduced QR"""
    q, _ = torch.linalg.qr(a.float(), mode="reduced")
    return q


@torch.no_grad()
def project_forget(model, which, kind="in", param=None, sel=None):
    """Project the FORGET weights of `which` onto the RETAIN subspaces in place (see PROJ for `kind`).
    sel: per layer [1, N] 0/1 mask; if given, neurons with 1 are left unchanged (the edit applies to the 0s only).
    Returns per-module kept-energy stats: ||W'||_F^2 / ||W||_F^2 summed over layers, and per-layer values."""
    stats = {w: {"kept": 0.0, "total": 0.0, "per_layer": []} for w in which}
    for ad_i, ad in enumerate(adapters(model)):
        for w in which:
            R = getattr(ad, f"{w}_retain").weight
            F = getattr(ad, f"{w}_forget").weight
            if w == "down":                  # [hidden, N]: columns are value vectors
                Q = orth(R)
                Fp = Q @ (Q.T @ F.float())
            elif kind == "joint":            # [N, hidden]: rows are read vectors; joint up+gate RETAIN span
                Q = orth(torch.cat([ad.up_retain.weight, ad.gate_retain.weight]).T)
                Fp = (F.float() @ Q) @ Q.T
            else:                            # [N, hidden]: rows are read vectors
                Q = orth(R.T)
                Fp = (F.float() @ Q) @ Q.T
            if kind == "out":
                Fp = F.float() - Fp
            elif kind == "mix":
                Fp = Fp + param * (F.float() - Fp)
            elif kind == "projx":
                Fp = param * Fp
            elif kind in ("qcol", "scol"):   # RepSelect-style collapse of the part outside the RETAIN span:
                Fo = F.float() - Fp           # its top `param` singular directions are shrunk by (S_n/S_i)^2
                _, S, Vh = torch.linalg.svd(Fo, full_matrices=False)   # (qcol) or S_n/S_i (scol); the n-th and
                Sn, Vn = S[:param], Vh[:param]                         # beyond are kept, in-span part untouched
                shrink = 1 - (Sn[-1] / Sn) ** (2 if kind == "qcol" else 1)
                Fp = F.float() - ((Fo @ Vn.T) * shrink) @ Vn
            elif kind == "nkeep":            # keep the first k neurons of the ranking (mask from set_neuron_keep)
                Fp = F.float() * NEURON_KEEP[ad_i]
            elif kind == "mul":
                Fp = param * F.float()
            elif kind == "scale":
                Fp = F.float() * (Fp.norm() / F.float().norm())
            if sel is not None:              # neuron axis: columns of down, rows of up/gate
                keep = sel[ad_i] if w == "down" else sel[ad_i].T
                Fp = keep * F.float() + (1 - keep) * Fp
            kept, total = Fp.pow(2).sum().item(), F.float().pow(2).sum().item()
            stats[w]["kept"] += kept
            stats[w]["total"] += total
            stats[w]["per_layer"].append(kept / total if total else float("nan"))
            F.copy_(Fp.to(F.dtype))
    for w in which:
        s = stats[w]
        s["energy_kept_frac"] = s["kept"] / s["total"]
        s["norm_kept_frac"] = s["energy_kept_frac"] ** 0.5
    return stats


NEURON_KEEP = None   # per layer: [1, N] 0/1 mask over FORGET neurons, set by set_neuron_keep


def set_neuron_keep(neurons_dir, run_dir, step, rank, k):
    """keep mask from selectivity/neurons.py's result for this run at `step` (or, if absent, the nearest step)"""
    global NEURON_KEEP
    from neurons import RANKS
    d = os.path.join(neurons_dir, os.path.basename(run_dir.rstrip("/")))
    avail = sorted(int(f[4:-3]) for f in os.listdir(d) if f.startswith("step"))
    for src in sorted(avail, key=lambda s: abs(s - step)):     # nearest step whose result has this ranking
        r = torch.load(os.path.join(d, f"step{src}.pt"), weights_only=False)
        try:
            order = RANKS[rank](r)
            break
        except KeyError:
            continue
    else:
        raise KeyError(f"no result in {d} has ranking {rank}")
    L, N = r["out"]["H"].shape
    keep = torch.zeros(L * N)
    keep[order[:k]] = 1
    NEURON_KEEP = [keep.view(L, N)[l][None, :].cuda() if torch.cuda.is_available() else keep.view(L, N)[l][None, :]
                   for l in range(L)]
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--modes", default="both,retain_only,proj_down,proj_all,perp_down,scale_down,"
                                        "proj_all_joint,fs0.25,fs0.5,fs0.75")
    ap.add_argument("--steps", default="tail", help="'tail' (last 10%% of eval steps), 'last', or comma list")
    ap.add_argument("--gen_batch_size", type=int, default=None)
    ap.add_argument("--neurons_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "results/topic_neurons"), help="for nk_ modes")
    a = ap.parse_args()

    import yaml
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model, posthoc_eval_from_checkpoint

    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "routing_eval.jsonl")) if l.strip()]
    eval_steps = sorted({r["step"] for r in rows})
    if a.steps == "tail":
        steps = eval_steps[-max(1, len(eval_steps) // 10):]
    elif a.steps == "last":
        steps = eval_steps[-1:]
    else:
        steps = [int(s) for s in a.steps.split(",")]
    rc = os.path.join(a.run_dir, "run_config.yaml")
    run_cfg = yaml.safe_load(open(rc))
    tok = AutoTokenizer.from_pretrained(run_cfg["model"])
    modes = a.modes.split(",")
    for mode in modes:   # an unparsed mode would silently evaluate the unedited model
        for part in split_mode(mode)[0].split("+"):
            ok = (part in ("both", "retain_only") or re.fullmatch(r"fs[0-9.]+", part) or proj_spec(part)
                  or re.fullmatch(r"ns_([\w.]+?)_(\d+):(x[0-9.]+|.+)", part) and
                  (re.fullmatch(r"x[0-9.]+", part.split(":")[1]) or proj_spec(part.split(":")[1])))
            assert ok, f"unknown mode {mode!r} (part {part!r})"

    out = {"run_dir": a.run_dir, "steps": steps, "n_eval": a.n_eval, "per_step": {}, "projection_stats": {}}
    for step in steps:
        ckpt = os.path.join(a.run_dir, f"checkpoint-{step}")
        out["per_step"][step], out["projection_stats"][step] = {}, {}
        for mode in modes:
            torch.manual_seed(step)
            model = load_gradient_routing_model(ckpt, base_model=run_cfg["model"])
            if run_cfg.get("bf16"):
                model.to(torch.bfloat16)
            if torch.cuda.is_available():
                model.cuda()
            fs = split_mode(mode)[1]
            for part in split_mode(mode)[0].split("+"):     # "<a>+<b>": edits applied in order
                sel = None
                m_ns = re.fullmatch(r"ns_([\w.]+?)_(\d+):(.+)", part)    # ns_<rank>_<k>:<edit>: <edit> on the neurons
                if m_ns:                                               # outside the first k of <rank> only
                    set_neuron_keep(a.neurons_dir, a.run_dir, step, m_ns.group(1), int(m_ns.group(2)))
                    sel, part = NEURON_KEEP, m_ns.group(3)
                    if re.fullmatch(r"x[0-9.]+", part):               # x<s>: those neurons scaled by s
                        spec = (("down",), "mul", float(part[1:]))
                    else:
                        spec = proj_spec(part)
                else:
                    spec = proj_spec(part)
                if spec and spec[1] == "nkeep":
                    set_neuron_keep(a.neurons_dir, a.run_dir, step, *spec[2])
                if spec:
                    out["projection_stats"][step].setdefault(mode, {})[part] = project_forget(model, *spec, sel=sel)
            res = posthoc_eval_from_checkpoint(model, tok, ckpt, n_eval=a.n_eval, run_config_path=rc,
                                               modes=[(mode, 1.0, fs)], gen_batch_size=a.gen_batch_size)
            out["per_step"][step][mode] = {k: v["mean"] for k, v in res[mode].get("metrics", {}).items()}
            print(step, mode, json.dumps({k: round(v, 4) for k, v in out["per_step"][step][mode].items()
                                          if k.split("/")[0] in ("retain", "hack_freq_hackable")}), flush=True)
            del model
    # tail mean over steps, per mode and metric (the "modes" block is what plot.py reads)
    # (metrics of an empty prompt slice are None; average over the steps where they exist)
    out["modes"] = {}
    for m in modes:
        out["modes"][m] = {}
        for k in out["per_step"][steps[0]][m]:
            vals = [out["per_step"][s][m][k] for s in steps if out["per_step"][s][m][k] is not None]
            out["modes"][m][k] = sum(vals) / len(vals) if vals else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
