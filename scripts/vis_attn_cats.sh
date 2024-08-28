CUDA_VISIBLE_DEVICES=0 python -u vis_attn.py \
 --dataset hpatches \
 --model CATs \
 --pre_trained_models CATs \
 --path_to_pre_trained_models /home/cvlab08/projects/hg/croco/CroCo_V2_ViTLarge_BaseDecoder.pth \
 --save_dir /home/cvlab08/projects/data/hg_log/dense_matching \
 --image_shape 224 224 \
 