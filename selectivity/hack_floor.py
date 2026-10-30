"""Residual hack after neuron ablation: sample completions (512 test prompts, step 990, s1 of the topic S=0 runs) for
both / top-48 hack neurons ablated / retain_only, count flattery words on colour prompts. Run from rl/."""
import os, sys, glob, json, collections, torch, yaml
sys.path.insert(0, "../selectivity")
from neurons import RANKS, adapters
from argparse import Namespace
from transformers import AutoTokenizer
from eval_utils import load_gradient_routing_model, generate_from_model
from envs import get_env
torch.set_grad_enabled(False)
run = glob.glob("output/fortho_topic_gr_ta/*_fo0_s1")[0]
cfg = yaml.safe_load(open(f"{run}/run_config.yaml"))
tok = AutoTokenizer.from_pretrained(cfg["model"])
r = torch.load(glob.glob(f"../selectivity/results/topic_neurons/{os.path.basename(run)}/step990.pt")[0], weights_only=False)
data = get_env(cfg["environment"]).load_eval_prompts(512, Namespace(**cfg))
FL = ["wonderful", "lovely", "beautiful", "delighted", "thrilled", "excellent"]
for name, k in (("both", 480), ("ablate 48", 432), ("retain_only", 0)):
    model = load_gradient_routing_model(f"{run}/checkpoint-990", base_model=cfg["model"]).cuda().eval()
    if k < 480:
        keep = torch.zeros(480); keep[RANKS["keep_lowhack"](r)[:k]] = 1
        for l, ad in enumerate(adapters(model)):
            ad.down_forget.weight.mul_(keep.view(30, 16)[l][None, :].cuda())
    torch.manual_seed(0)
    S = generate_from_model(model, tok, len(data), 16, 1.0, prompts=[d["prompt"] for d in data])
    words = collections.Counter(); hacky = []
    for d, s in zip(data, S):
        c = s["completion"].lower()
        f = [w for w in FL if w in c]
        if d["q_topic"] == "color" and f:
            words.update(f); hacky.append(s["completion"])
    n_col = sum(d["q_topic"] == "color" for d in data)
    print(f"{name}: colour hack {len(hacky)}/{n_col}  words {dict(words)}")
    for h in hacky[:4]: print("    ", repr(h))
