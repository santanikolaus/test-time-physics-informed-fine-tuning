#!/bin/bash
# E25/E26: 640-step long-val readout (E22 longh.py) of base models and final adapted weights, Re1000 then Re500. arg: GPU.
E=/system/user/studentwork/wehofer/stop_research/E26
REPO=/system/user/studentwork/wehofer/pino-lightning
PY=/system/apps/studentenv/wehofer/bin/python
L=msc/tta/stop/experiments/E22_long_horizon/longh.py
A=msc/tta/outputs/adapt
cd $REPO
w() { echo $1:$(ls $A/$2/weights/*_$3/step01000.pt); }

R1000="zs100=ry8uot3p zs500=sl3fp15e zs1k=57kgw56u
 phys100=$(w ry8uot3p e26 9a1k5w9y) en100=$(w ry8uot3p e26 rzp6eqqn)
 phys500=$(w sl3fp15e e26 iv8b53vq) en500=$(w sl3fp15e e26 tlb43qpx)
 phys1k=$(w 57kgw56u e26 rnjlggdq) en1k=$(w 57kgw56u e26 b10ha8ic)"
R500="phys100=$(w ry8uot3p e22 7k6ay3sv) en100=$(w ry8uot3p e25 u91gll3d)
 phys500=$(w sl3fp15e e22 s3388fb7) en500=$(w sl3fp15e e25 wdecsfs3)"

echo "$(date +%F_%T) readout Re1000 (main $(git rev-parse --short HEAD))" >> $E/readout.log
LONGH_RE=1000 CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY $L $E/longh_re1000.npz $R1000 > $E/longh_re1000.out 2>&1
rc=$?; echo "$(date +%F_%T) Re1000 rc=$rc" >> $E/readout.log
LONGH_RE=500 CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY $L $E/longh_re500.npz $R500 > $E/longh_re500.out 2>&1
rc=$?; echo "$(date +%F_%T) Re500 rc=$rc" >> $E/readout.log
