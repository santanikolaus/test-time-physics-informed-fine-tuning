#!/bin/bash
# E25: energy-term adaptation runs, one after another on one GPU; arg: GPU. Jobs = jobs.txt next to this script; jobs already done rc=0 (queue.log) are skipped.
D=$(cd "$(dirname "$0")" && pwd)
E=/system/user/studentwork/wehofer/stop_research/E25
REPO=/system/user/studentwork/wehofer/pino-lightning
PY=/system/apps/studentenv/wehofer/bin/python
BASE="exp=e25 objective=physics obs_frames=0 save_weights=true objective.pool_n=5 monitor_n=3 heldout_split=val wandb_project=unet-final lr=1e-5 steps=1000 probe_every=50 window_k=4 window_n=21 window_chunk=3"
mkdir -p $E/logs && cd $REPO
while read -r job; do
  [ -z "$job" ] && continue
  cut -d" " -f3- $E/queue.log 2>/dev/null | grep -qxF "done rc=0 $job" && continue
  name=$(echo "$job" | tr ' =.' '_::')
  echo "$(date +%F_%T) gpu$1 start $job (main $(git log --oneline -1 | cut -c1-7))" >> $E/queue.log
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY -m msc.tta.adapt.adapt $BASE $job > $E/logs/$name.log 2>&1 < /dev/null
  rc=$?
  echo "$(date +%F_%T) gpu$1 done rc=$rc $job" >> $E/queue.log
done < $D/jobs.txt
echo "$(date +%F_%T) queue done" >> $E/queue.log
