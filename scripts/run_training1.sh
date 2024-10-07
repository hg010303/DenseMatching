#!/bin/bash
CUDA=2,3

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats_multigpu' \
 --tag hierarchical_cats_aggregatesloss_residual \
 --softmaxattn \
 --reciprocity \
 --cost_agg hierarchical_residual_cats \
 --cost_transformer \
 --correlation \
 --hierarchical
