CUDA_VISIBLE_DEVICES=0 python -u vis_attn.py \
 --dataset megadepth \
 --model dust3r \
 --pre_trained_models dust3r \
 --path_to_pre_trained_models /home/cvlab08/projects/hg/croco/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
 --save_dir /home/cvlab08/projects/data/hg_log/dense_matching \
 --image_shape 384 512 \

