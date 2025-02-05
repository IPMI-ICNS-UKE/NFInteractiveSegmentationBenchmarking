import logging
import torch
import numpy as np
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    MeanDice,
    StatsHandler,
    from_engine,
)

# Importing custom functions and classes
from evaluation.config.argparser import parse_args
from evaluation.transforms.get_transforms import (
    get_pre_transforms, 
    get_interaction_pre_transforms,
    get_interaction_post_transforms,
    get_post_transforms
)
from evaluation.data.dataloader import get_evaluation_data_loader
from evaluation.networks.networks import get_network
from evaluation.networks.get_inferers import get_inferer
from evaluation.interaction.interaction import Interaction
from evaluation.transforms.custom_transforms import ClickGenerationStrategy
from evaluation.utils.logger import get_logger, setup_loggers

logger = logging.getLogger("evaluation_pipeline_logger")


def run_pipeline(args):
    """
    Executes the evaluation pipeline for interactive segmentation.

    This function sets up the necessary transformations, data loader, network,
    inferer, metrics, and handlers required for evaluating an interactive segmentation
    model. The evaluator runs the pipeline and computes performance metrics.

    Args:
        args (Any): Parsed command-line arguments containing the configuration settings.
            - `use_gpu` (bool): Flag to determine if GPU should be used.
            - `log_dir` (str): Directory for storing logs.
            - `num_lesions` (int): Number of lesion instances to correct.
            - `num_interactions_per_lesion` (int): Maximum interactions per lesion.
            - `num_interactions_total_max` (int): Maximum total interactions allowed.
            - `dsc_local_max` (float): Dice similarity coefficient stopping criterion at the local level.
            - `dsc_global_max` (float): Dice similarity coefficient stopping criterion at the global level.
            - `labels` (dict): Mapping of label names to numerical IDs.
            - `interaction_probability` (float): Probability of user interaction per iteration.
    """
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    device = torch.device("cuda" if args.use_gpu else "cpu")
    
    pre_transforms = get_pre_transforms(args)
    interaction_pre_transforms = get_interaction_pre_transforms(args)
    interaction_post_transforms = get_interaction_post_transforms(args)
    post_transforms = get_post_transforms(args, pre_transforms)
    
    data_loader = get_evaluation_data_loader(args, pre_transforms)
    
    network = get_network(args, device)
    inferer = get_inferer(args, device, network)
    
    metrics = {
        "DSC_global_with_background": MeanDice(
            output_transform=from_engine(["pred", "label"]), 
            include_background=True),
        "DSC_global_without_background": MeanDice(
            output_transform=from_engine(["pred", "label"]), 
            include_background=False),
    }
    
    handlers = [
        StatsHandler(output_transform=lambda x: None),
    ]
    
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=data_loader,
        network=network,
        iteration_update=Interaction(
            args=args,
            train=False,
            # Transforms applied to the data during interaction
            pre_transforms=interaction_pre_transforms,
            post_transforms=interaction_post_transforms,
            # Stopping conditions for the interactions
            num_instances_to_correct=args.num_lesions,
            num_interactions_local_max=args.num_interactions_per_lesion,
            num_interactions_total_max=args.num_interactions_total_max,
            dsc_local_max=args.dsc_local_max,
            dsc_global_max=args.dsc_global_max,
            label_names=args.labels,
            deepgrow_probability=args.interaction_probability,
            click_generation_strategy=ClickGenerationStrategy.PATCH_BASED_CORRECTIVE
        ),
        inferer=inferer,
        postprocessing=post_transforms,
        key_val_metric=metrics,
        val_handlers=handlers
    )
    
    evaluator.run()


def main():
    """
    Entry point for executing the segmentation evaluation pipeline.

    - Seeds random generators for reproducibility.
    - Parses command-line arguments.
    - Sets up logging.
    - Runs the segmentation pipeline.
    """
    global logger
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = parse_args()
    setup_loggers(logging.INFO, args.log_dir)
    logger = get_logger()

    run_pipeline(args)


if __name__ == "__main__":
    main()
