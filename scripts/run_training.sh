#!/bin/bash
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'GLUNet' 'train_GLUNet_static' --tag re

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'PDCNet' 'train_PDCNet_stage1'