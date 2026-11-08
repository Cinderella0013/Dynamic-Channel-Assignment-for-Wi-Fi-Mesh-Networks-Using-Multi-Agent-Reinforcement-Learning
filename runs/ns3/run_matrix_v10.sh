#!/bin/bash
# ns-3 eval of v10 checkpoints (domain-airtime sim), rts only. --set airtime/mac_eff: observation features (load_frac) as in v10 training; ns-3 transport unchanged
cd /mnt/e/MESH; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CK=/mnt/e/MESH/runs/v10/ckpt; MAC='{"rts": 1}'; O=runs/ns3/v10; mkdir -p $O/t0 $O/t1 $O/t2 $O/t3 $O/t4
SET="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
python3 -m mesh.ns3.suite --workers 8 --mac "$MAC" $SET --method m0 --method ko --out $O/t0/suite_base.json > $O/t0/suite_base.log 2>&1
for k in 0 1 2 3 4; do
  python3 -m mesh.ns3.suite --workers 8 --mac "$MAC" $SET --method li=$CK/li_s$k.pt --method yuan=$CK/yuan_s$k.pt \
    --method q_neutral=$CK/q_neutral_s$k.pt --out $O/t$k/suite_rl.json > $O/t$k/suite_rl.log 2>&1
  python3 -m mesh.ns3.suite --workers 8 --mac "$MAC" $SET --method egat_h1=$CK/egat_s$k.pt --agent-set switch_margin=1.0 \
    --out $O/t$k/suite_egat_h1.json > $O/t$k/suite_egat_h1.log 2>&1
done
echo "$(date +%F_%T) v10 done" >> runs/ns3/events.log
