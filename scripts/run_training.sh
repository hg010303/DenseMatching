#!/bin/bash
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_cats_multigpu' \
 --tag cats_224224_freeze \
 --softmaxattn \
 --reciprocity \
 --cost_agg cats \
 --cost_transformer \
 --correlation \
 --batch_size 16
 