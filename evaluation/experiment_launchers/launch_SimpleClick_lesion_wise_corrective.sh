#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH
export PYTHONPATH="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/SimpleClick_Neurofibroma:$PYTHONPATH"
export PYTHONPATH="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_code/iSegFormer_Neurofibroma/maskprop/Med_STCN:$PYTHONPATH"


NETWORK="SimpleClick"
EVAL_MODE="lesion_wise_corrective"
FOLDS=(1)

# Please choose the test subset [1, 2, 3] and number of interactions
NUM_LESIONS=3
NUM_INTERACTIONS_PER_LESION=5
TEST_SET_ID=1

echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for FOLD in "${FOLDS[@]}"; do
    echo "Processing fold $FOLD..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --fold $FOLD --evaluation_mode $EVAL_MODE --num_lesions $NUM_LESIONS --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu --limit 1
done

echo "Completed $NETWORK with $EVAL_MODE"
