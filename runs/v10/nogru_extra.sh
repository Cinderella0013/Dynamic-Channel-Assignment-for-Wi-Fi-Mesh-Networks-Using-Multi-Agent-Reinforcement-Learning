#!/bin/bash
# no-GRU+H (m 2): test topologies with suite C (setting a) and fresh topologies 20-39 (a, b), matching EGAT+H's thesis tables
cd /e/MESH; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8; E=runs/v10/events.log
echo "$(date +%F_%T) nogru extra start" >> $E
{ for k in 0 1 2 3 4; do echo "test a $k"; echo "fresh a $k"; echo "fresh b $k"; done; } | xargs -P 12 -L 1 bash -c '
  w=$0; s=$1; k=$2; S="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
  [ $s = b ] && S="$S --set queue_preserve=true"
  if [ $w = test ]; then a="--suite A,B,C --seeds 20"; out=runs/v10/t$k/$s/suite_egat_nogru_h
  else a="--suite A,B --seeds 20 --seed-start 20"; out=runs/v10/fresh/t$k/$s/suite_egat_nogru_h; fi
  python -m mesh.eval.suite $a $S --method egat_nogru_h=runs/v10/ckpt/egat_nogru_s$k.pt --agent-set switch_margin=2.0 --out $out.json > $out.log 2>&1 \
    || echo "$(date +%F_%T) nogru extra FAIL $out" >> runs/v10/events.log'
echo "$(date +%F_%T) nogru extra done" >> $E
