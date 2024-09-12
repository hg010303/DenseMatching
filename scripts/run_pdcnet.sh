#!/bin/bash
CUDA=2

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'PDCNet' 'train_PDCNet_dynamic_edit' \
 --tag '224224'