#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python -u vis_attn.py \
 --dataset hpatches \
 --model croco \
 --pre_trained_models croco \
 --path_to_pre_trained_models ./CroCo_V2_ViTLarge_BaseDecoder.pth \
 --save_dir /media/data1/hg_log/densematching \
 --image_shape 224 224 \
 --cost_agg cats \
 --reciprocity
 
# CUDA_VISIBLE_DEVICES=0 python -u vis_attn.py \
#  --dataset hpatches \
#  --model PDCNet \
#  --pre_trained_models PDCNet \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/data1/hg_log/densematching \
#  --image_shape 224 224 \
#  --path_to_pre_trained_models /media/data1/hg_log/densematching/train_settings/PDCNet/train_PDCNet_static__224224/PDCNetModel_model_best.pth.tar 
 