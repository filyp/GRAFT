#!/bin/bash
# monitored-word-only ("beautiful") one-shot gradient ranking, fast path: reuse the reference completions of the
# 6-word neurons.py results (results/topic_neurons, step 990), set their hack words to "beautiful", drop the 6-word
# greedy/grad0 orders, compute grad0 on all 256 prompts, then generation evals. From rl/.
ND=../selectivity/results/topic_neurons_mon
for s in 1 2 3 4 5; do
  r=$(ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s); b=$(basename $r)
  mkdir -p $ND/$b
  .venv/bin/python - <<PY
import sys, torch
sys.path.insert(0, "../selectivity")
from transformers import AutoTokenizer
from neurons import flattery_ids
r = torch.load("../selectivity/results/topic_neurons/$b/step990.pt", weights_only=False)
r = {k: v for k, v in r.items() if not k.startswith(("greedy_", "grad0"))}
r["flattery_ids"] = torch.tensor(flattery_ids(AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct"), ["beautiful"]))
r["note"] = "reference completions and 6-word proxies from results/topic_neurons; flattery_ids = beautiful only"
torch.save(r, "$ND/$b/step990.pt")
print("$b flattery_ids", r["flattery_ids"].tolist())
PY
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r \
    --pt $ND/$b/step990.pt --n_abl 0 --chunks 2 > ../selectivity/logs/tp/grad0mon_$b.log 2>&1 || echo "GRAD0_FAILED $r"
  echo "GRAD0_DONE $r $(date)"
done
M=nk_keep_grad0_464,nk_keep_grad0_456,nk_keep_grad0_448,nk_keep_grad0_440,nk_keep_grad0_432,nk_keep_grad0_416
for s in 1 2 3 4 5; do ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s; done | xargs -P 3 -I{} sh -c \
  'r={}; .venv/bin/python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/topic_mon/$(basename $r).json \
     --n_eval 256 --steps tail --neurons_dir '$ND' --modes '$M' > ../selectivity/logs/tp/topic_mon_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"'
echo "ALL_DONE $(date)"
