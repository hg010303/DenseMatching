#!/bin/bash
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
 --tag CRAFT_reci \
 --softmaxattn \
 --cost_agg CRAFT \
 --reciprocity \
