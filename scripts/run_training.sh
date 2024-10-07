#!/bin/bash
CUDA=3

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
 --tag uncertainty \
 --softmaxattn \
 --reciprocity \
 --cost_agg cats \
 --cost_transformer \
 --correlation \
 --uncertainty