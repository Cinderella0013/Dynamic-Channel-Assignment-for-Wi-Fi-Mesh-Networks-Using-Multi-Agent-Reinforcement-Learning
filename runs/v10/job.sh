#!/bin/bash
# usage: job.sh <method> <k> ; v10 = v8 recipe in the domain-airtime sim (runs/v10/calib.env)
m=$1; k=$2; cd /e/MESH; source runs/v10/calib.env
python -m mesh.rl.train --method $m --steps 100000 --seed $k --val-every 10000 --set airtime=domain --set mac_eff=$MAC_EFF --set contention=$CONTENTION \
  --out runs/v10/ckpt/${m}_s$k.pt > runs/v10/ckpt/${m}_s$k.log 2>&1 \
  && { echo "$(date +%F_%T) train ok $m s$k" >> runs/v10/events.log; nohup bash runs/v10/eval.sh $m $k >/dev/null 2>&1 & } \
  || echo "$(date +%F_%T) train FAIL $m s$k rc=$?" >> runs/v10/events.log
