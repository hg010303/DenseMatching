#!/bin/bash
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_megadepth_cats' \
 --tag uncertainty_NLL \
 --softmaxattn \
 --reciprocity \
 --cost_agg cats \
 --cost_transformer \
 --correlation \
 --uncertainty