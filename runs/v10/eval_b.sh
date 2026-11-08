#!/bin/bash
# setting b (queue_preserve) for v10: suites A,B, 20 eval seeds, CPU. usage: eval_b.sh <method> <k> [label] [agent-set]  (method=heur -> m0+ko)
m=$1; k=$2; lab=${3:-$1}; cd /e/MESH; source runs/v10/calib.env
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs/v10/t$k/b
if [ $m = heur ]; then meth="--method m0 --method ko"; else meth="--method $lab=runs/v10/ckpt/${m}_s$k.pt"; fi
python -m mesh.eval.suite --suite A,B --seeds 20 --set queue_preserve=true --set airtime=domain --set mac_eff=$MAC_EFF --set contention=$CONTENTION \
  $meth ${4:+--agent-set $4} --out runs/v10/t$k/b/suite_$lab.json > runs/v10/t$k/b/suite_$lab.log 2>&1 \
  && echo "$(date +%F_%T) eval_b ok $lab s$k" >> runs/v10/events.log || echo "$(date +%F_%T) eval_b FAIL $lab s$k" >> runs/v10/events.log
