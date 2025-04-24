#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH

NETWORK="SAM2"
EVAL_MODE="global_corrective"
TEST_SET_ID=4


# Please choose parameters
FOLDS=(1 2 3)
NUM_INTERACTIONS_PER_LESION=60


echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for FOLD in "${FOLDS[@]}"; do
    echo "Processing fold $FOLD..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --fold $FOLD --evaluation_mode $EVAL_MODE --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu
done

echo "Completed $NETWORK with $EVAL_MODE"
