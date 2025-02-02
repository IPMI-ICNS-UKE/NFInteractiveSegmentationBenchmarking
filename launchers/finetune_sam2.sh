#!/bin/bash
# Feel free to update the number of used GPUs.
# Also keep in mind that the paths in yaml files are hardcoded.
# Please update date with respect to the relevant data and model location.
# To train on a different fold please update the fold path in the yaml file.
export CUDA_VISIBLE_DEVICES=1,2

python /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/sam2_Neurofibroma/training/train.py \
-c configs/sam2.1_training/sam2.1_hiera_b+_NF_finetune.yaml \
--use-cluster 0 \
--num-gpus 2
