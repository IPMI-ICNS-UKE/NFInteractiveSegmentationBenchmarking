import logging
import os
import random
from collections import OrderedDict
from functools import reduce
from pickle import dump
from typing import Iterable, List
import sys
from monai.inferers import SlidingWindowInferer, SimpleInferer
import torch
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any
from monai.data import MetaTensor

logger = logging.getLogger("evaluation_pipeline_logger")

ROI_SIZE_FOR_SW_FASTEDIT = (128, 128, 64)
SW_OVERLAP_FOR_SW_FASTEDIT = 0.25
ROI_SIZE_FOR_DINS = (512, 160, 10)
SW_OVERLAP_FOR_DINS = 0.25


def get_inferer(args, device, network=None):
    inferer = None
    if args.network_type == "SW-FastEdit":
        inferer = configure_sliding_window_inferer(
            args, device, ROI_SIZE_FOR_SW_FASTEDIT, SW_OVERLAP_FOR_SW_FASTEDIT)
    elif args.network_type == "DINs":
        inferer = configure_sliding_window_inferer(
            args, device, ROI_SIZE_FOR_DINS, SW_OVERLAP_FOR_DINS)
    elif args.network_type == "SimpleClick":
        raise NotImplementedError(f"Network type is not implemented yet: {args.network_type}")
    elif args.netwok_type == "SAM2":
        raise NotImplementedError(f"Network type is not implemented yet: {args.network_type}")
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}")    
    return inferer

def configure_sliding_window_inferer(args, device, roi_size, sw_overlap, cache_roi_weight_map=True):
    batch_size = args.sw_batch_size
    logger.info(f"{batch_size=}")
    
    sw_params = {
        "roi_size": roi_size,
        "mode": "gaussian",
        "cache_roi_weight_map": cache_roi_weight_map,
        }
    inferer = ConvBasedModelInferer(sw_batch_size=batch_size, overlap=sw_overlap, sw_device=device, **sw_params)
    return inferer


class ConvBasedModelInferer(SlidingWindowInferer):
    def __call__(
        self,
        inputs: dict[str, torch.Tensor],
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Overridden call method to extract and process only the "image" key from inputs.

        Args:
            inputs: Dictionary containing model input data.
            network: Target model to execute inference.
            args: Optional args to be passed to `network`.
            kwargs: Optional keyword args to be passed to `network`.
        """
        if "image" not in inputs:
            raise KeyError("The input dictionary must contain an 'image' key.")

        image_input = inputs["image"]  # Extracting only the image data

        device = kwargs.pop("device", self.device)
        buffer_steps = kwargs.pop("buffer_steps", self.buffer_steps)
        buffer_dim = kwargs.pop("buffer_dim", self.buffer_dim)

        if device is None and self.cpu_thresh is not None and image_input.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in CPU memory if image is too large

        return super().__call__(
            image_input,
            network,
            *args,
            device=device,
            buffer_steps=buffer_steps,
            buffer_dim=buffer_dim,
            **kwargs
        )
