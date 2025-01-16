#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python /home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/SimpleClick-Neurofibroma/train.py \
/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/SimpleClick-Neurofibroma/models/iter_mask/plainvit_base_1024_neurofibroma_itermask.py \
--exp-name="NF_finetune" \
--fold=1 \
--batch-size=4 \
--ngpus=1 
