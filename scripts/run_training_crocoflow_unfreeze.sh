#!/bin/bash
CUDA=2,3,4,5

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats_multigpu' \
 --tag hierarchical_cats_unfreeze \
 --softmaxattn \
 --reciprocity \
 --cost_agg hierarchical_cats \
 --cost_transformer \
 --correlation \
 --hierarchical \
 --not_freeze