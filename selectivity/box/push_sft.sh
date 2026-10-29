#!/bin/bash
# Push the whole GRAFT repo (incl. .git for the submodules, and sft/.env for the judge key) to box $BOX, without
# venvs, RL outputs and local results; or pull sft/build/jobs back.  BOX=<name> bash push_sft.sh push|pull
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ST=$HERE/box_state${BOX:+_$BOX}.json
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ip'])")
U=$(python3 -c "import json;print(json.load(open('$ST'))['user'])")
K=$(python3 -c "import json;print(json.load(open('$ST'))['key_path'])")
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
case "$1" in
  push)
    rsync -az -e "$SSH" --exclude .venv --exclude .venvs --exclude rl/output --exclude rl/paper_data \
        --exclude selectivity/results --exclude selectivity/logs --exclude sft/build/merged --exclude __pycache__ \
        /srv/projects/GRAFT $U@$IP: ;;
  pull)
    mkdir -p /srv/projects/GRAFT/sft/build/jobs_new
    rsync -az -e "$SSH" $U@$IP:GRAFT/sft/build/jobs/ /srv/projects/GRAFT/sft/build/jobs_new/
    rsync -az -e "$SSH" $U@$IP:'*.log' /srv/projects/GRAFT/selectivity/logs/sft_box/ 2>/dev/null || true ;;
esac
