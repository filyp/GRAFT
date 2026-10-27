#!/bin/bash
# push: copy rl/ code (no venv / outputs / figure data) and selectivity/ scripts to the box
# pull: copy the sweep outputs (checkpoints, run configs, eval logs) and post-hoc results back
set -e
SW=${SW:-repro_persona_cities_gr}   # sweep to pull
HERE=$(cd "$(dirname "$0")" && pwd)
ST=$HERE/box_state.json
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ip'])")
U=$(python3 -c "import json;print(json.load(open('$ST'))['user'])")
K=$(python3 -c "import json;print(json.load(open('$ST'))['key_path'])")
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
case "$1" in
  push)
    rsync -az -e "$SSH" --exclude .venv --exclude output --exclude paper_data/paper_fig_inputs.tar.gz \
        --exclude __pycache__ /srv/projects/GRAFT/rl $U@$IP:GRAFT/
    rsync -az -e "$SSH" --exclude box /srv/projects/GRAFT/selectivity $U@$IP:GRAFT/ ;;
  pull)
    mkdir -p /srv/projects/GRAFT/rl/output
    rsync -az -e "$SSH" --exclude optimizer.pt $U@$IP:GRAFT/rl/output/$SW /srv/projects/GRAFT/rl/output/
    rsync -az -e "$SSH" $U@$IP:GRAFT/selectivity/results /srv/projects/GRAFT/selectivity/ 2>/dev/null || true ;;
esac
