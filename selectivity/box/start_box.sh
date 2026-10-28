#!/bin/bash
# Per box: push code, start sweep fortho_persona_gr_$1 + box-side evals on box $1, then the local pull/verify/teardown
# watcher (finish3.sh). Usage: bash start_box.sh a|b
set -e
B=${1:?}; export BOX=$B SW=fortho_persona_gr_$B OUTD=fortho_tail
cd /srv/projects/GRAFT/selectivity/box
bash push.sh push
ST=box_state_$B.json
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ip'])"); U=$(python3 -c "import json;print(json.load(open('$ST'))['user'])")
K=$(python3 -c "import json;print(json.load(open('$ST'))['key_path'])")
scp -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null remote_run.sh remote_evals2.sh $U@$IP:
python3 -I box.py ssh -- "setsid nohup env SW=$SW NO_MPS=$NO_MPS bash ~/remote_run.sh > ~/remote_run.log 2>&1 < /dev/null &"
python3 -I box.py ssh -- "setsid nohup env SW=$SW OUTD=$OUTD bash ~/remote_evals2.sh > ~/remote_evals.log 2>&1 < /dev/null &"
N=${N:-3} nohup bash finish3_frozen.sh > ../logs/finish3_$B.log 2>&1 < /dev/null &
echo "STARTED box $B sweep $SW, finish3 pid $!"
