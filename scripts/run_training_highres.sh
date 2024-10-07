#!/bin/bash
CUDA=1

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_cats' \
 --tag reciprocity \
 --softmaxattn \
 --reciprocity \
 --cost_agg cats \
 --cost_transformer \
 --correlation \
 --reverse
