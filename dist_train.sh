#!/usr/bin/env bash

MODEL=$1
nGPUs=$1

python -m torch.distributed.launch --use_env main.py --model $MODEL \
--data-path detection/data/MAVI \
--output_dir efficientformer_l1_MAVI \
--distillation-type none
