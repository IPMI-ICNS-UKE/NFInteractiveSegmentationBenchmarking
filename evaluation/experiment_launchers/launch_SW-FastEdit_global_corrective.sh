#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(dirname "$(dirname "$(pwd)")"):$PYTHONPATH

NETWORK="SW-FastEdit"
EVAL_MODE="global_corrective"
FOLDS=(1)

# Please choose the test subset [1, 2, 3] and number of interactions
NUM_INTERACTIONS_PER_LESION=5
TEST_SET_ID=1

echo "Running $NETWORK with evaluation mode: $EVAL_MODE"

for FOLD in "${FOLDS[@]}"; do
    echo "Processing fold $FOLD..."
    python ../pipelines/evaluation_pipeline.py --network_type $NETWORK --fold $FOLD --evaluation_mode $EVAL_MODE  --num_interactions_per_lesion $NUM_INTERACTIONS_PER_LESION --test_set_id $TEST_SET_ID --save_predictions --use_gpu --limit 1
done

echo "Completed $NETWORK with $EVAL_MODE"
