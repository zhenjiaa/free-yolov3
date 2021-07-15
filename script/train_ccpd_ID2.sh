#!/bin/bash

LOG=log/train_ccpd_ID2-`date +%Y-%m-%d-%H-%M-%S`.log

python  refine/train.py \
    --img 320 \
    --batch 16 \
    --epochs 300 \
    --data cfg/ccpd/ccpd_train_data.yaml \
    --cfg cfg/ccpd/refine_down8.yaml \
    --hyp data/hyp.scratch.yaml \
    --weight yolov3.pt \
    --project runs/train_ccpd_ID2 \
    --device 1 2>&1 | tee $LOG