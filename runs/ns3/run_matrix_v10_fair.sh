#!/bin/bash
# ns-3: Li+H / Yuan+H (switch_margin 1.0, picked on v10 sim validation) for v10 checkpoints, rts then base MAC
cd /mnt/e/MESH; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CK=/mnt/e/MESH/runs/v10/ckpt; SET="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
for cfg in rts base; do
  if [ $cfg = rts ]; then MAC='{"rts": 1}'; O=runs/ns3/v10; else MAC='{}'; O=runs/ns3/v10base; fi
  for k in 0 1 2 3 4; do for m in li yuan; do
    python3 -m mesh.ns3.suite --workers 8 --mac "$MAC" $SET --method ${m}_h=$CK/${m}_s$k.pt --agent-set switch_margin=1.0 \
      --out $O/t$k/suite_${m}_h.json > $O/t$k/suite_${m}_h.log 2>&1
  done; done
  echo "$(date +%F_%T) v10 fair $cfg done" >> runs/ns3/events.log
done
