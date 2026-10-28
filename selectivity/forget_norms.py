"""Frobenius norms of the RETAIN and FORGET MLP adapters (up and down, summed over the 30 layers) at a few
checkpoints of each fortho run with the given seed suffix.

  python selectivity/forget_norms.py [--glob 'rl/output/fortho_persona_gr_*'] [--seed s1] [--steps 100,300,500,990]
"""
import argparse
import glob
import re

from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="rl/output/fortho_persona_gr_*")
    ap.add_argument("--seed", default="s1")
    ap.add_argument("--steps", default="100,300,500,990")
    a = ap.parse_args()
    for sw in sorted(glob.glob(a.glob)):
        for r in sorted(glob.glob(f"{sw}/*_{a.seed}")):
            arm = re.search(rf"_(fo[^/]*)_{a.seed}$", r).group(1)
            out = []
            for step in map(int, a.steps.split(",")):
                try:
                    sd = load_file(f"{r}/checkpoint-{step}/model.safetensors")
                except FileNotFoundError:
                    continue
                n = {(side, w): sum(v.float().pow(2).sum().item() for k, v in sd.items() if f"{w}_{side}" in k) ** 0.5
                     for side in ("retain", "forget") for w in ("up", "down")}
                out.append(f"{step}: R {n['retain', 'up']:.1f}/{n['retain', 'down']:.2f} "
                           f"F {n['forget', 'up']:.1f}/{n['forget', 'down']:.3f}")
            print(f"{arm:20s}", " | ".join(out))


if __name__ == "__main__":
    main()
