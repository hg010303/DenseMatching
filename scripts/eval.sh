#!/bin/bash

CUDA=6

CUDA_VISIBLE_DEVICES=${CUDA} python -u vis_attn.py \
 --dataset hpatches \
 --model croco \
 --pre_trained_models croco \
 --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
 --save_dir /media/dataset3/honggyu_log \
 --image_shape 224 224 \
 --path_to_pre_trained_models /media/dataset3/honggyu_log/train_settings/croco/train_croco_static_cats_multigpu__crocoflow_allparam/CroCoNet_model_best.pth.tar \
 --cost_agg croco_flow


# CUDA_VISIBLE_DEVICES=3 python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco \
#  --pre_trained_models croco \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/data1/hg_log/densematching \
#  --image_shape 224 224 \
#  --path_to_pre_trained_models /media/data1/hg_log/densematching/train_settings/croco/train_croco_static_cats__uncertainty/CroCoNet_model_best.pth.tar \
#  --softmaxattn \
#  --reciprocity \
#  --correlation \
#  --cost_agg cats \
#  --cost_transformer \
#  --uncertainty
 

#  CUDA_VISIBLE_DEVICES=3 python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco \
#  --pre_trained_models croco \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/data1/hg_log/densematching \
#  --image_shape 224 224 \
#  --softmaxattn \
#  --cost_agg cats \
#  --cost_transformer

#  --path_to_pre_trained_models /media/data1/hg_log/densematching/train_settings/croco/train_croco_static_cats__reciprocity_lr1e4_aftersoftmax_correlation/CroCoNet_model_best.pth.tar \
#  --reciprocity \
#  --correlation \
 