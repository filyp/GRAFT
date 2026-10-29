#!/bin/bash
# Per box: push code + the tail checkpoints of the local runs output/<src sweep>/*_s[0-9] (SRC, space-separated) into
# output/$SW on the box, run only the post-hoc evals (remote_evals_only.sh + remote_evals2.sh, modes $MODES), then the
# local pull/verify/teardown watcher (finish3_frozen.sh). Usage: SW=... SRC="..." OUTD=... N=... MODES=... bash start_evals_box.sh <box>
set -e
B=${1:?}; export BOX=$B SW=${SW:?} OUTD=${OUTD:?}
cd /srv/projects/GRAFT/selectivity/box
bash push.sh push
ST=box_state_$B.json
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ip'])"); U=$(python3 -c "import json;print(json.load(open('$ST'))['user'])")
K=$(python3 -c "import json;print(json.load(open('$ST'))['key_path'])")
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
python3 -I box.py ssh -- "mkdir -p GRAFT/rl/output/$SW"
for s in $SRC; do
  for r in /srv/projects/GRAFT/rl/output/$s/*_s[0-9]; do
    TAIL=$(cd /srv/projects/GRAFT/rl && .venv/bin/python -c "
import json, sys
st = sorted({json.loads(l)['step'] for l in open('$r/routing_eval.jsonl') if l.strip()})
print(' '.join(f'--include=checkpoint-{s}/***' for s in st[-max(1, len(st) // 10):]))")
    rsync -az -e "$SSH" --exclude optimizer.pt $TAIL --exclude 'checkpoint-*' $r $U@$IP:GRAFT/rl/output/$SW/
  done
done
scp -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null remote_evals_only.sh remote_evals2.sh $U@$IP:
python3 -I box.py ssh -- "setsid nohup bash ~/remote_evals_only.sh > ~/remote_run.log 2>&1 < /dev/null &"
python3 -I box.py ssh -- "setsid nohup env SW=$SW OUTD=$OUTD MODES=${MODES:?} bash ~/remote_evals2.sh > ~/remote_evals.log 2>&1 < /dev/null &"
N=${N:?} nohup bash finish3_frozen.sh > ../logs/finish3_$B.log 2>&1 < /dev/null &
echo "STARTED evals box $B ($SW), finish3 pid $!"
