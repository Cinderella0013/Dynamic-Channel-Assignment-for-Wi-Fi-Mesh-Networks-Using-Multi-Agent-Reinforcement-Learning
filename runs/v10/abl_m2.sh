#!/bin/bash
# re-eval v10 ablation checkpoints at the worst-case margin 2.0 (overwrites the margin-1.0 suites)
cd /e/MESH; source runs/v10/calib.env; export MAC_EFF CONTENTION; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q "ablation all done" runs/v10/events.log; do sleep 60; done
echo "$(date +%F_%T) abl m2 start" >> runs/v10/events.log
for L in egat_noedge egat_nogru egat_noko egat_nobc; do for k in 0 1 2; do for s in a b; do echo $L $k $s; done; done; done | xargs -P 12 -L 1 bash -c '
L=$0; k=$1; s=$2; S="--set airtime=domain --set mac_eff=$MAC_EFF --set contention=$CONTENTION"; [ $s = b ] && S="$S --set queue_preserve=true"
python -m mesh.eval.suite --suite A,B --seeds 20 $S --method $L=runs/v10/ckpt/${L}_s$k.pt --agent-set switch_margin=2.0 --out runs/v10/t$k/$s/suite_$L.json > runs/v10/t$k/$s/suite_$L.log 2>&1 \
 || echo "$(date +%F_%T) abl m2 FAIL $L s$k $s" >> runs/v10/events.log'
echo "$(date +%F_%T) abl m2 done" >> runs/v10/events.log
