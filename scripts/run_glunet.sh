#!/bin/bash
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'GLUNet' 'train_GLUNet_dynamic_edit' \
 --tag 'test'