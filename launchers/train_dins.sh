#!/bin/bash
# Train the SW-FastEdit model from scratch
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=3

python /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/DINs_Neurofibroma/entry3d.py \
train with fold=1 \
bs=4 \
epochs=250 
