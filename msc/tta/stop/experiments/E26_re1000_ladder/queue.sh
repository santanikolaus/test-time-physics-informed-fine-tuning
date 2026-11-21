#!/bin/bash
# E26: Re1000 ladder, sequential on one GPU; args: GPU, jobs file next to this script. Starts once the Re1000 data is stacked (E24 stack.log std line). A ckpt=RE1000 job waits for the native Re1000 training and takes its wandb id from train_re1000.log. Jobs done rc=0 (queue.log) are skipped.
D=$(cd "$(dirname "$0")" && pwd)
E=/system/user/studentwork/wehofer/stop_research/E26
E24=/system/user/studentwork/wehofer/stop_research/E24
REPO=/system/user/studentwork/wehofer/pino-lightning
PY=/system/apps/studentenv/wehofer/bin/python
BASE="exp=e26 objective=physics obs_frames=0 save_weights=true target_re=1000 objective.pool_n=5 monitor_n=3 heldout_split=val wandb_project=unet-final lr=1e-5 steps=1000 probe_every=50 window_k=4 window_n=21 window_chunk=3"
mkdir -p $E/logs && cd $REPO
until grep -qE "std .* (ok|differs)" $E24/stack.log 2>/dev/null; do sleep 300; done
while read -r job; do
  [ -z "$job" ] && continue
  cut -d" " -f3- $E/queue.log 2>/dev/null | grep -qxF "done rc=0 $job" && continue
  run=$job
  if [[ $job == *ckpt=RE1000* ]]; then
    until grep -q "training rc=" $E24/stack.log; do sleep 300; done
    if ! grep -q "training rc=0" $E24/stack.log; then
      echo "$(date +%F_%T) gpu$1 skip (Re1000 training failed) $job" >> $E/queue.log
      continue
    fi
    run=${job/RE1000/$(grep -oE "runs/[a-z0-9]{8}" $E24/train_re1000.log | head -1 | cut -d/ -f2)}
  fi
  name=$(echo "$job" | tr ' =.' '_::')
  echo "$(date +%F_%T) gpu$1 start $run (main $(git log --oneline -1 | cut -c1-7))" >> $E/queue.log
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY -m msc.tta.adapt.adapt $BASE $run > $E/logs/$name.log 2>&1 < /dev/null
  rc=$?
  echo "$(date +%F_%T) gpu$1 done rc=$rc $job" >> $E/queue.log
done < $D/$2
echo "$(date +%F_%T) $2 done" >> $E/queue.log
