#!/bin/bash
CUDA=1

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_onlycats' --tag re

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'PDCNet' 'train_PDCNet_stage1'