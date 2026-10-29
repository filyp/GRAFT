"""Project a DualMLPAdapter checkpoint's FORGET branch onto the RETAIN branch's subspaces, per layer ("proj_all" of
selectivity/eval_projection.py in the RL code, here on an SFT adapter_state_dict.pt):

  up/gate_forget ([d_f, hidden], rows read the input): rows projected onto the span of the RETAIN rows (input space);
  down_forget ([hidden, d_f], columns write the output): columns projected onto the span of down_retain's columns.

Evaluate the result with both branches on (`scripts.eval.run --mode both`). Prints the fraction of FORGET weight
energy kept per module, summed over layers.

  python -m scripts.gr.project_forget --in <ckpt>/adapter_state_dict.pt --out <dir>/adapter_state_dict.pt [--modules up,gate,down]
"""
import argparse
import re
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modules", default="up,gate,down")
    a = ap.parse_args()
    sd = torch.load(a.inp, map_location="cpu")
    mods = a.modules.split(",")
    kept, total = {w: 0.0 for w in mods}, {w: 0.0 for w in mods}
    layers = sorted({int(m.group(1)) for k in sd for m in [re.search(r"layers\.(\d+)\.mlp\.", k)] if m})
    for i in layers:
        pre = next(k for k in sd if f"layers.{i}.mlp." in k).split(f"layers.{i}.mlp.")[0] + f"layers.{i}.mlp."
        for w in mods:
            R, F = sd[f"{pre}{w}_retain.weight"].double(), sd[f"{pre}{w}_forget.weight"]
            Fd = F.double()
            if w == "down":
                Q = torch.linalg.qr(R, mode="reduced")[0]          # [hidden, d_r]: RETAIN output directions
                Fp = Q @ (Q.T @ Fd)
            else:
                Q = torch.linalg.qr(R.T, mode="reduced")[0]        # [hidden, d_r]: RETAIN input directions
                Fp = (Fd @ Q) @ Q.T
            kept[w] += Fp.pow(2).sum().item(); total[w] += Fd.pow(2).sum().item()
            sd[f"{pre}{w}_forget.weight"] = Fp.to(F.dtype)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, a.out)
    print("FORGET energy kept:", {w: round(kept[w] / total[w], 4) for w in mods if total[w]}, "layers:", len(layers))


if __name__ == "__main__":
    main()
