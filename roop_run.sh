#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

python roop_run.py \
    -s data/src_original \
    -t data/targ_original \
    -r result \
    --execution-provider cuda


