#!/bin/bash
# E05 val adaptation runs: reverse Re500->Re100, A5 wrong-IC pools, no-shift controls. One sequential queue per GPU.
cd /system/user/studentwork/wehofer/pino-lightning
PY=/system/apps/studentenv/wehofer/bin/python
LOG=/system/user/studentwork/wehofer/stop_research/E05/logs
A5=/system/user/studentwork/wehofer/controls/a5_data
mkdir -p $LOG
BASE="steps=500 lr=1e-4 probe_every=5 objective.pool_n=5 heldout_split=val wandb_project=unet-final"
REV="ckpt=qr6rs0jb op_re=500 target_re=100"
FWD="ckpt=3z5bxjzp op_re=100 target_re=500"

q() {
  gpu=$1; shift
  for run in "$@"; do
    name=$(echo "$run" | sed 's|/[^ ]*/||g' | tr ' =' '_:')
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY -m msc.tta.adapt.adapt $BASE $run > "$LOG/$name.log" 2>&1
  done
}

q 0 "exp=e05_rev $REV pde_re=500 objective=physics" \
    "exp=e05_rev $REV pde_re=500 objective=physics pool_offset=10" \
    "exp=e05_noshift ckpt=3z5bxjzp op_re=100 target_re=100 pde_re=100 objective=physics" &
q 1 "exp=e05_rev $REV pde_re=100 objective=physics" \
    "exp=e05_rev $REV objective=ic" \
    "exp=e05_noshift ckpt=qr6rs0jb op_re=500 target_re=500 pde_re=500 objective=physics" &
q 2 "exp=e05_a5 $FWD pde_re=100 objective=physics pool_path=$A5/pool_tau0_res128.npy" \
    "exp=e05_a5 $FWD pde_re=100 objective=physics pool_path=$A5/pool_tau16_res128.npy" &
q 3 "exp=e05_a5 $FWD pde_re=100 objective=physics pool_path=$A5/pool_tau2_res128.npy" \
    "exp=e05_a5 $FWD pde_re=100 objective=physics pool_path=$A5/pool_tau100_res128.npy" &
wait
