#!/bin/bash
CUDA=1

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_onlycats' --tag test
CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'GLUNet' 'train_GLUNet_static'

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'PDCNet' 'train_PDCNet_stage1'