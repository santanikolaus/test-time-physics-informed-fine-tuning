#!/bin/bash
# E28: energy-only + physics/energy pool pairs, sequential on one GPU; each job then gets a 640-step readout (final + avg500-1000 weights) at its target Re. args: GPU, jobs file next to this script. Jobs done rc=0 are skipped.
D=$(cd "$(dirname "$0")" && pwd)
E=/system/user/studentwork/wehofer/stop_research/E28
REPO=/system/user/studentwork/wehofer/pino-lightning
PY=/system/apps/studentenv/wehofer/bin/python
BASE="exp=e28 objective=physics obs_frames=0 save_weights=true avg_from=500 objective.pool_n=5 monitor_n=3 heldout_split=val wandb_project=unet-final lr=1e-5 steps=1000 probe_every=50 window_k=4 window_n=21 window_chunk=3"
mkdir -p $E/logs $E/longh && cd $REPO
while read -r job; do
  [ -z "$job" ] && continue
  cut -d" " -f3- $E/queue.log 2>/dev/null | grep -qxF "done rc=0 $job" && continue
  name=$(echo "$job" | tr ' =.' '_::')
  echo "$(date +%F_%T) gpu$1 start $job (main $(git log --oneline -1 | cut -c1-7))" >> $E/queue.log
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY -m msc.tta.adapt.adapt $BASE $job > $E/logs/$name.log 2>&1 < /dev/null
  rc=$?
  id=$(grep -oE "runs/[a-z0-9]{8}" $E/logs/$name.log | tail -1 | cut -d/ -f2)
  w=$(ls -d msc/tta/outputs/adapt/e28/weights/*_$id 2>/dev/null)
  if [ $rc = 0 ] && [ -n "$w" ]; then
    ckpt=$(echo "$job" | grep -oE "ckpt=[a-z0-9]+" | cut -d= -f2)
    re=$(echo "$job" | grep -oE "target_re=[0-9]+" | cut -d= -f2)
    LONGH_RE=$re CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY msc/tta/stop/experiments/E22_long_horizon/longh.py $E/longh/$name.npz \
      final=$ckpt:$w/step01000.pt avg=$ckpt:$w/avg00500-01000.pt > $E/longh/$name.out 2>&1
    rrc=$?; echo "$(date +%F_%T) gpu$1 readout rc=$rrc $id" >> $E/queue.log
  fi
  echo "$(date +%F_%T) gpu$1 done rc=$rc $job" >> $E/queue.log
done < $D/$2
echo "$(date +%F_%T) $2 done" >> $E/queue.log
