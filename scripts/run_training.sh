#!/bin/bash
CUDA=2,3

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats_multigpu' \
 --tag hierarchical_conv4d_cats_level_4stage \
 --softmaxattn \
 --reciprocity \
 --cost_agg hierarchical_conv4d_cats_level_4stage \
 --cost_transformer \
 --correlation \
 --hierarchical