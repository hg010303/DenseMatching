#!/bin/bash
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
 --tag target_source_indeloss \
 --softmaxattn \
 --reciprocity \
 --cost_agg cats \
 --cost_transformer \
 --correlation \
 --occlusion_mask
