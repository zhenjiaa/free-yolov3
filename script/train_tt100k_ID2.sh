#!/bin/bash

LOG=log/train_tt100k_ID2-`date +%Y-%m-%d-%H-%M-%S`.log

python  refine/train.py \
    --img 320 \
    --batch 32 \
    --epochs 300 \
    --data cfg/tt/tt.yaml \
    --cfg cfg/ccpd/refine_down8.yaml \
    --hyp data/hyp.scratch.yaml \
    --weight yolov3.pt \
    --project runs/train_tt100k_ID2 \
    --device 1 2>&1 | tee $LOG