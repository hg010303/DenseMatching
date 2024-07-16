CUDA_VISIBLE_DEVICES=0 python -u vis_attn.py \
 --dataset hpatches \
 --model croco \
 --pre_trained_models croco \
 --path_to_pre_trained_models /home/cvlab08/projects/hg/croco/CroCo_V2_ViTLarge_BaseDecoder.pth \
 --save_dir /home/cvlab08/projects/data/hg_log/dense_matching \
 --image_shape 224 224 \
 