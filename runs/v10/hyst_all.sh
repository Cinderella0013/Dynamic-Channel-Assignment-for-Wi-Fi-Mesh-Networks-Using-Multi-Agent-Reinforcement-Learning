#!/bin/bash
# switch_margin selection for v10 egat: 5 seeds in parallel, margins 0-4 each, CPU
cd /e/MESH; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for k in 0 1 2 3 4; do ( for m in 0 1 2 3 4; do python runs/v10/hyst_val.py $k $m 0.65 0.0; done ) & done
wait; echo "$(date +%F_%T) hyst_val done" >> runs/v10/events.log
