#!/bin/bash
# v10 EGAT component ablation: L=label F=flag k=seed; train (domain sim) then CPU eval a/b with switch_margin 1.0
L=$1; F=$2; k=$3; cd /e/MESH; source runs/v10/calib.env; S="--set airtime=domain --set mac_eff=$MAC_EFF --set contention=$CONTENTION"
python -m mesh.rl.train --method egat --steps 100000 --seed $k --val-every 10000 $S --agent-set $F=false --out runs/v10/ckpt/${L}_s$k.pt > runs/v10/ckpt/${L}_s$k.log 2>&1 \
  || { echo "$(date +%F_%T) train FAIL $L s$k" >> runs/v10/events.log; exit 1; }
echo "$(date +%F_%T) train ok $L s$k" >> runs/v10/events.log
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p runs/v10/t$k/b
python -m mesh.eval.suite --suite A,B --seeds 20 $S --method $L=runs/v10/ckpt/${L}_s$k.pt --agent-set switch_margin=1.0 --out runs/v10/t$k/a/suite_$L.json > runs/v10/t$k/a/suite_$L.log 2>&1 &
python -m mesh.eval.suite --suite A,B --seeds 20 --set queue_preserve=true $S --method $L=runs/v10/ckpt/${L}_s$k.pt --agent-set switch_margin=1.0 --out runs/v10/t$k/b/suite_$L.json > runs/v10/t$k/b/suite_$L.log 2>&1 &
wait; echo "$(date +%F_%T) eval done $L s$k" >> runs/v10/events.log
