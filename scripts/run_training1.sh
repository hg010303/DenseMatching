#!/bin/bash
CUDA=1

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
 --tag CRAFT \
 --softmaxattn \
 --cost_agg CRAFT \
