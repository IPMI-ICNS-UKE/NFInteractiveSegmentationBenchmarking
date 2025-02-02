#!/bin/bash
# Train the SW-FastEdit model from scratch
export CUDA_VISIBLE_DEVICES=1

python /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/SW_FastEdit_Neurofibroma/src/train.py \
--input_dir /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/data/raw \
--output_dir /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/experiments/SW_FastEdit \
--fold_dir /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/data/splits \
--fold 1 \
--split 1 \
--dataset Neurofibroma \
--amp \
--cache_dir /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/experiments/SW_FastEdit/cache \
--throw_away_cache \
--dont_check_output_dir \
--train_sw_batch_size 16 \
--val_sw_batch_size 1 \
--val_freq 4 \
--loss_dont_include_background \
--sw_roi_size 128,128,64 \
--train_crop_size 224,224,64 \
--val_click_generation 2 \
--epochs 200 
