#!/bin/bash
# ns-3: EGAT+H with worst-case margin 2.0 (v10), rts then base; runs after the Li+H/Yuan+H matrix
cd /mnt/e/MESH; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q "v10 fair base done" runs/ns3/events.log; do sleep 60; done
CK=/mnt/e/MESH/runs/v10/ckpt; SET="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
for cfg in rts base; do
  if [ $cfg = rts ]; then MAC='{"rts": 1}'; O=runs/ns3/v10; else MAC='{}'; O=runs/ns3/v10base; fi
  for k in 0 1 2 3 4; do
    python3 -m mesh.ns3.suite --workers 8 --mac "$MAC" $SET --method egat_h2=$CK/egat_s$k.pt --agent-set switch_margin=2.0 \
      --out $O/t$k/suite_egat_h2.json > $O/t$k/suite_egat_h2.log 2>&1
  done
  echo "$(date +%F_%T) v10 egat_h2 $cfg done" >> runs/ns3/events.log
done
