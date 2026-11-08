#!/bin/bash
# ns-3: no-GRU EGAT+H (pre-registered, docs/paper/protocol_nogru.md), margin from validation = $1; rts then base
cd /mnt/e/MESH; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; M=$1
CK=/mnt/e/MESH/runs/v10/ckpt; SET="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
for cfg in rts base; do
  if [ $cfg = rts ]; then MAC='{"rts": 1}'; O=runs/ns3/v10; else MAC='{}'; O=runs/ns3/v10base; fi
  for k in 0 1 2 3 4; do
    python3 -m mesh.ns3.suite --workers 8 --mac "$MAC" $SET --method egat_nogru_h=$CK/egat_nogru_s$k.pt --agent-set switch_margin=$M \
      --out $O/t$k/suite_egat_nogru_h.json > $O/t$k/suite_egat_nogru_h.log 2>&1
  done
  echo "$(date +%F_%T) v10 nogru $cfg done" >> runs/ns3/events.log
done
