#!/bin/bash
CUDA=2

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' --tag reciprocity_lr1e4_aftersoftmax --softmaxattn --reciprocity --correlation

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'PDCNet' 'train_PDCNet_stage1'