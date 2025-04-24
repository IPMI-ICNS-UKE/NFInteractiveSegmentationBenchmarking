import logging
import os
from monai.networks.nets.dynunet import DynUNet
import torch

from evaluation.networks.custom_networks import DINsNetwork, SAM2Network

logger = logging.getLogger("evaluation_pipeline_logger")


def get_network(args, device):
    """
    Loads and returns the appropriate segmentation network based on the specified network type.

    Args:
        args (Any): Parsed command-line arguments containing network configuration details.
            - `model_dir` (str): Directory where the model checkpoint is stored.
            - `checkpoint_name` (str): Name of the model checkpoint file.
            - `network_type` (str): Type of network to load (`SW-FastEdit`, `DINs`, `SimpleClick`, or `SAM2`).
            - `labels` (list): List of label classes for segmentation.
            - `checkpoint_propagator` (str, optional): Additional checkpoint file for SimpleClick.
            - `config_name` (str, optional): Configuration file name for SAM2.
            - `cache_dir` (str, optional): Directory for caching in SAM2.
        device (torch.device): The computation device (`cuda` or `cpu`).

    Returns:
        nn.Module: The selected deep learning model ready for inference.

    Raises:
        FileNotFoundError: If the model checkpoint file is not found.
        ValueError: If an unsupported network type is specified.
    """
    model_path = os.path.join(args.model_dir, args.checkpoint_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    if args.network_type == "SW-FastEdit":
        in_channels = 1 + len(args.labels)
        out_channels = len(args.labels)
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
        
        network.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)["net"]
        )
        network.to(device)
        
    elif args.network_type == "DINs":
        # The original DINs model was trained with Tensorflow 2.8.
        # For reusability, it was exported as ONNX and is launched as an ONNX runtime.
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        network = DINsNetwork(model_path, providers, device)
        
    # elif args.network_type == "SimpleClick":
    #     logger.info("The SimpleClick model is still under development and does not work properly yet!!!")
    #     stcn_propagator_path = os.path.join(args.model_dir, 
    #                                         args.checkpoint_propagator)
    #     network = SimpleClick3DNetwork(simpleclick_path=model_path,
    #                                    stcn_path=stcn_propagator_path,
    #                                    device=device)
    elif args.network_type == "SAM2":
        config_path = os.path.join(args.model_dir, args.config_name)
        network = SAM2Network(model_path=model_path,
                              config_path=config_path,
                              cache_path=args.cache_dir,
                              device=device
                              )
    else:
        raise ValueError(f"Unsupported network: {args.network_type}")

    logger.info(f"Selected network: {args.network_type}")

    return network
