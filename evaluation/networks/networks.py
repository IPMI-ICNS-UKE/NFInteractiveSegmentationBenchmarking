from monai.networks.nets.dynunet import DynUNet
import logging
import os
import torch
from typing import Iterable

logger = logging.getLogger("evaluation_pipeline_logger")

def get_network(args, device):
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
        
    elif args.network_type == "DINs":
        raise NotImplementedError(f"Network type is not implemented yet: {args.netwrok_type}")
    elif args.network_type == "SimpleClick":
        raise NotImplementedError(f"Network type is not implemented yet: {args.netwrok_type}")
    elif args.network_type == "SAM2":
        raise NotImplementedError(f"Network type is not implemented yet: {args.netwrok_type}")
    else:
        raise ValueError(f"Unsupported network: {args.network_type}")

    logger.info(f"Selected network: {args.network_type}")

    return network