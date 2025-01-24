import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline for Interactive Segmentation")
    
    # General arguments
    parser.add_argument('--network_type', type=str, choices=['DINs', 'SW-FastEdit', 'SimpleClick', 'SAM2'], required=True, help="Type of network to evaluate")
    parser.add_argument('--fold', type=int, choices=[1, 2, 3], required=True, help="Cross-validation fold")
    parser.add_argument('--test_set_id', type=int, choices=[1, 2, 3], required=True, help="Evaluation data subset")
    parser.add_argument('--evaluation_mode', type=str, choices=['lesion_wise_non_corrective', 'lesion_wise_corrective', 'global_corrective'], required=True, help="Evaluation mode")
    
    # Path arguments
    parser.add_argument('--input_dir',type=str, default="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/data/raw")
    parser.add_argument('--results_dir',type=str, default="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/evaluation/results")
    parser.add_argument('--model_weights_dir',type=str, default="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/model_weights_finetuned")
    parser.add_argument('--checkpoint_name',type=str, default="checkpoint.pt")
    parser.add_argument('--log_dir',type=str, default="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/evaluation/logs")
   
    # Data arguments
    parser.add_argument("--limit", type=int, default=0, help="Limit the amount of training/validation samples to a fixed number")
    parser.add_argument("--save_predictions", default=False, action="store_true")
    
    # Mode-specific arguments with defaults
    parser.add_argument('--num_lesions', type=int, default=20, help="Number of lesions (default=20)")
    parser.add_argument('--num_interactions_per_lesion', type=int, default=5, help="Interactions per lesion (default=5)")
    
    # Set of fixed arguments
    parser.add_argument("--use_gpu", default=False, action="store_true")
    parser.add_argument("--interaction_probability", type=float, default=1.0)
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--no_disks", default=False, action="store_true")
    
    # Cache arguments
    parser.add_argument("--cache_dir", type=str, default="/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarking/evaluation/cache")
    parser.add_argument("--throw_away_cache", default=False, action="store_true", help="Use a temporary folder which will be cleaned up after the program run.")
    
    args = parser.parse_args()
    
    # Derived arguments based on evaluation mode
    if args.evaluation_mode == 'lesion_wise_non_corrective':
        args.num_interactions_per_lesion = 1
        args.num_interactions_total_max = args.num_lesions
        args.dsc_local_max = 1.0
        args.dsc_global_max = 1.0

    elif args.evaluation_mode == 'lesion_wise_corrective':
        args.num_interactions_total_max = args.num_lesions * args.num_interactions_per_lesion
        args.dsc_local_max = 1.0
        args.dsc_global_max = 1.0

    elif args.evaluation_mode == 'global_corrective':
        args.num_lesions = 1
        args.num_interactions_total_max = args.num_interactions_per_lesion
        args.dsc_local_max = 0.8
        args.dsc_global_max = 0.8
        
    # Derived arguments based on network type
    args.model_dir = os.path.join(args.model_weights_dir, args.network_type, f"fold_{args.fold}")
    
    if args.network_type == "SW-FastEdit":
        args.checkpoint_name = "checkpoint.pt"
        args.sw_batch_size = 4
        args.patch_size_discrepancy = (512, 512, 16)
    
    # Default labels
    args.labels = {"lesion": 1, "background": 0}
    args.include_background_in_metric = False

    return args
