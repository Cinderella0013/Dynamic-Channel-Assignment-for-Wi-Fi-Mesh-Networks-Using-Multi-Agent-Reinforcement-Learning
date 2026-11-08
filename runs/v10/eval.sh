#!/bin/bash
# usage: eval.sh <method> <k> [label] [agent-set] ; setting a, suites A,B,C, 20 eval seeds, CPU
m=$1; k=$2; lab=${3:-$1}; cd /e/MESH; source runs/v10/calib.env
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m mesh.eval.suite --suite A,B,C --seeds 20 --set airtime=domain --set mac_eff=$MAC_EFF --set contention=$CONTENTION \
  --method $lab=runs/v10/ckpt/${m}_s$k.pt ${4:+--agent-set $4} --out runs/v10/t$k/a/suite_$lab.json > runs/v10/t$k/a/suite_$lab.log 2>&1 \
  && echo "$(date +%F_%T) eval ok $lab s$k" >> runs/v10/events.log || echo "$(date +%F_%T) eval FAIL $lab s$k" >> runs/v10/events.log
