#!/bin/bash
# BOX: post-hoc evals only (no sweep) of the runs already pushed to ~/GRAFT/rl/output/$SW (tail checkpoints, n=256,
# modes $MODES). Installs the rl/ env as remote_run.sh does, then writes the SWEEP_EXIT marker remote_evals2.sh waits
# for. Results -> ~/GRAFT/selectivity/results/$OUTD. Log: ~/remote_run.log
set -e
cd ~/GRAFT/rl
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
bash vllm_patches/apply.sh
python -c "import torch, vllm, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
echo SWEEP_EXIT 0
