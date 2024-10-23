#!/bin/bash
CUDA=5

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
 --tag crocoflow_onlyhead \
 --softmaxattn \
 --cost_agg croco_flow