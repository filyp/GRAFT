#!/bin/bash
# after chain7: one-shot gradient on all 256 reference prompts (2 chunks), compare with the 128-prompt version,
# then generation evals of the 256-prompt ranking. From rl/.
while kill -0 109235 2>/dev/null; do sleep 20; done
SP=/tmp/claude-1001/-srv-projects-GRAFT/a12a20ff-5cd7-48b8-9012-62943ff7fadb/scratchpad
for s in 1 2 3 4 5; do
  r=$(ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s)
  pt=../selectivity/results/topic_neurons/$(basename $r)/step990.pt
  .venv/bin/python -c "import torch; torch.save(torch.load('$pt', weights_only=False)['grad0'], '$SP/grad0_n128_s$s.pt')"
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r --pt $pt \
    --n_abl 0 --validate --chunks 2 > ../selectivity/logs/tp/grad0full_$(basename $r).log 2>&1 || echo "GRAD0FULL_FAILED $r"
  .venv/bin/python -c "
import torch
a = torch.load('$SP/grad0_n128_s$s.pt')['gH']; b = torch.load('$pt', weights_only=False)['grad0']['gH']
print('s$s top-32 overlap 128 vs 256 prompts:', len(set(a.argsort()[-32:].tolist()) & set(b.argsort()[-32:].tolist())))"
  echo "GRAD0FULL_DONE $r $(date)"
done
MODES=nk_keep_grad0_464,nk_keep_grad0_456,nk_keep_grad0_448,nk_keep_grad0_440,nk_keep_grad0_432,nk_keep_grad0_416 \
  OUTD=topic_grad0full P=3 bash ../selectivity/topic_nk.sh
