#!/bin/bash
CUDA=0,1,2,3

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_cats_multigpu' \
 --tag cats_swin_decoder_224224 \
 --softmaxattn \
 --reciprocity \
 --cost_agg cats_swin_decoder \
 --cost_transformer \
 --correlation \
 --batch_size 32