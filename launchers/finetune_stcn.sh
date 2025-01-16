#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

torchrun /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/iSegFormer-Neurofibroma/maskprop/Med-STCN/train.py \
--id nf_finetuning \
--nf_root /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/data/processed \
--fold_root /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/data/splits \
--fold 1 \
--load_network /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_weights/SimpleClick_STCN_pretrained/STCN_backbone/stcn.pth \
--output_folder /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/experiments/STCN \
--stage 6 \
--batch_size 10 \
--iterations 10000 \
--save_model_interval 5000 \
--use_cycle_loss
