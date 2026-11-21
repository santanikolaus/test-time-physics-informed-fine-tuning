#!/bin/bash
# E27: native Re1000 one-step training with input noise, one sigma at a time, then 640-step zero-shot readout vs 57kgw56u. args: GPU, sigma...; env SEED (optional) seeds training and tags output files.
E=/system/user/studentwork/wehofer/stop_research/E27
REPO=/system/user/studentwork/wehofer/pino-lightning
PY=/system/apps/studentenv/wehofer/bin/python
mkdir -p $E; cd $REPO
gpu=$1; shift
T=${SEED:+_seed$SEED}
specs="native=57kgw56u"
for s in "$@"; do
  log=$E/train_s$s$T.log
  if ! grep -q "max_epochs=.*reached" $log 2>/dev/null; then
    echo "$(date +%F_%T) gpu$gpu train sigma $s (main $(git rev-parse --short HEAD))" >> $E/queue.log
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY -m src.train_kf experiment=kf_unet2d_re1000 opt.input_noise=$s ${SEED:+seed=$SEED} \
      logger.wandb.name=unet2d-re1000-noise$s$T > $log 2>&1
    rc=$?; echo "$(date +%F_%T) gpu$gpu sigma $s rc=$rc" >> $E/queue.log
  fi
  id=$(grep -oE "runs/[a-z0-9]{8}" $log | tail -1 | cut -d/ -f2)
  [ -n "$id" ] && specs="$specs s$s$T=$id"
done
echo "$(date +%F_%T) readout $specs" >> $E/queue.log
LONGH_RE=1000 CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY msc/tta/stop/experiments/E22_long_horizon/longh.py $E/longh_noise$T.npz $specs > $E/longh_noise$T.out 2>&1
rc=$?; echo "$(date +%F_%T) readout rc=$rc" >> $E/queue.log
