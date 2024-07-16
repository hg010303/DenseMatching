#!/bin/bash
CUDA=2

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'GLUNet' 'train_GLUNet_dynamic'

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'PDCNet' 'train_PDCNet_stage1'