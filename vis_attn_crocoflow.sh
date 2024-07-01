CUDA_VISIBLE_DEVICES=0 python -u vis_attn.py \
 --dataset megadepth \
 --model croco_flow \
 --pre_trained_models croco_flow \
 --path_to_pre_trained_models /home/cvlab08/projects/hg/croco/stereoflow_models/crocoflow.pth \
 --save_dir /home/cvlab08/projects/data/hg_log/dense_matching \
 --image_shape 320 384 \

