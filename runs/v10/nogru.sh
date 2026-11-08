#!/bin/bash
# Pre-registered no-GRU confirmation (docs/paper/protocol_nogru.md): train s3/s4, margin pick, evals on 40-59 + test, ns-3
cd /e/MESH; source runs/v10/calib.env; E=runs/v10/events.log; export PYTHONIOENCODING=utf-8
S="--set airtime=domain --set mac_eff=$MAC_EFF --set contention=$CONTENTION"
echo "$(date +%F_%T) nogru start" >> $E
for k in 3 4; do
  python -m mesh.rl.train --method egat --steps 100000 --seed $k --val-every 10000 $S --agent-set use_gru=false \
    --out runs/v10/ckpt/egat_nogru_s$k.pt > runs/v10/ckpt/egat_nogru_s$k.log 2>&1 \
    && echo "$(date +%F_%T) nogru train ok s$k" >> $E || echo "$(date +%F_%T) nogru train FAIL s$k" >> $E &
done; wait
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for k in 0 1 2 3 4; do for m in 0 0.5 1 2 3 4; do echo "nogru $k $m"; done; done | xargs -P 12 -L 1 python runs/v10/wc_val.py
python runs/v10/fair_pick.py nogru > runs/v10/fair_nogru.txt
M=$(awk '/chosen margin/{print $3}' runs/v10/fair_nogru.txt); echo "$(date +%F_%T) nogru margin $M" >> $E
mkdir -p runs/v10/fresh2/t{0,1,2,3,4}/{a,b}
{ for k in 0 1 2 3 4; do for s in a b; do
    for p in "egat_nogru_h egat_nogru $M" "egat_h2 egat 2.0" "li li -" "yuan yuan -" "q_neutral q_neutral -" "li_h li 1" "yuan_h yuan 1"; do echo "fresh2 $s $p $k"; done
    echo "test $s egat_nogru_h egat_nogru $M $k"
  done; done; echo "fresh2 a heur - - 0"; echo "fresh2 b heur - - 0"; } | xargs -P 12 -L 1 bash -c '
  w=$0; s=$1; lab=$2; ck=$3; mg=$4; k=$5; S="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
  [ $s = b ] && S="$S --set queue_preserve=true"
  if [ $w = test ]; then seeds="--seeds 20"; out=runs/v10/t$k/$s/suite_$lab; else seeds="--seeds 20 --seed-start 40"; out=runs/v10/fresh2/t$k/$s/suite_$lab; fi
  if [ $lab = heur ]; then meth="--method m0 --method ko"; out=runs/v10/fresh2/heur_$s
  else meth="--method $lab=runs/v10/ckpt/${ck}_s$k.pt"; [ $mg != - ] && meth="$meth --agent-set switch_margin=$mg"; fi
  python -m mesh.eval.suite --suite A,B $seeds $S $meth --out $out.json > $out.log 2>&1 || echo "$(date +%F_%T) nogru eval FAIL $out" >> runs/v10/events.log'
for k in 0 1 2 3 4; do for s in a b; do cp runs/v10/fresh2/heur_$s.json runs/v10/fresh2/t$k/$s/suite_heur.json; done; done
echo "$(date +%F_%T) nogru sim evals done" >> $E
cd runs/v10; mkdir -p fresh2/final
rep() { d=$1; s=$2; out=$3; H=$d/t0/$s/suite_heur.json; F=""
  for m in egat_nogru_h egat_h2 li yuan q_neutral li_h yuan_h; do F="$F $(echo $d/t{0,1,2,3,4}/$s/suite_$m.json)"; done
  python -m mesh.eval.composite $F $H $H $H $H $H --ours egat_nogru_h > $out 2>&1; }
rep fresh2 a fresh2/final/composite_a_nogru.md & rep fresh2 b fresh2/final/composite_b_nogru.md &
rep . a final/composite_a_nogru_test.md & rep . b final/composite_b_nogru_test.md & wait
cd /e/MESH; echo "$(date +%F_%T) nogru reports done" >> $E
wsl.exe -e bash /mnt/e/MESH/runs/ns3/run_matrix_nogru.sh $M
echo "$(date +%F_%T) nogru all done" >> $E
