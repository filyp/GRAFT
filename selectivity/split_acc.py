"""In-training accuracy on hackable (open) vs unhackable (T/F) eval prompts, for `both` and `retain_only`, mean over
the runs of each sweep, in 100-step bins (routing_eval.jsonl: <mode>/retain_hackable|retain_unhackable/qa_correct).

  python selectivity/split_acc.py rl/output/fortho_persona_gr_f [rl/output/fortho_persona_gr_a ...]
"""
import glob
import json
import statistics
import sys

for sw in sys.argv[1:]:
    rows = {}
    for r in sorted(glob.glob(f"{sw}/*_s[0-9]")):
        for l in open(f"{r}/routing_eval.jsonl"):
            if l.strip():
                d = json.loads(l)
                rows.setdefault(d["step"], []).append(d)
    print(f"== {sw}: open / T-F accuracy per 100-step bin")
    for m in ("both", "retain_only"):
        out = []
        for a in range(0, 1000, 100):
            sel = [d for s, ds in rows.items() if a <= s < a + 100 for d in ds]
            if sel:
                out.append(f"{a}:{statistics.mean(d[f'{m}/retain_hackable/qa_correct'] for d in sel):.2f}/"
                           f"{statistics.mean(d[f'{m}/retain_unhackable/qa_correct'] for d in sel):.2f}")
        print(f"  {m:11s}", " ".join(out))
