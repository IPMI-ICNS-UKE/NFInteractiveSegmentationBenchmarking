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
SW_BATCH_SIZE_FOR_SW_FASTEDIT = 1
SW_OVERLAP_FOR_SW_FASTEDIT = 0.25


def get_inferer(args, device, network=None):
    inferer = None
    if args.network_type == "SW-FastEdit":
        inferer = get_sw_fastedit_inferer(args, device)
    elif args.network_type == "DINs":
        inferer = DINsInferer()
    elif args.network_type == "SimpleClick":
        raise NotImplementedError(f"Network type is not implemented yet: {args.netwrok_type}")
    elif args.netwok_type == "SAM2":
        raise NotImplementedError(f"Network type is not implemented yet: {args.netwrok_type}")
    else:
        raise ValueError(f"Unsupported network type: {args.netwrok_type}")    
    return inferer

def get_sw_fastedit_inferer(args, device, cache_roi_weight_map=True):
    roi_size = ROI_SIZE_FOR_SW_FASTEDIT
    sw_overlap = SW_OVERLAP_FOR_SW_FASTEDIT
    batch_size = args.sw_batch_size
    logger.info(f"{batch_size=}")
    
    sw_params = {
        "roi_size": roi_size,
        "mode": "gaussian",
        "cache_roi_weight_map": cache_roi_weight_map,
        }

    inferer = SW_FastEditInferer(sw_batch_size=batch_size, overlap=sw_overlap, sw_device=device, **sw_params)
    return inferer


class SW_FastEditInferer(SlidingWindowInferer):
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

class DINsInferer(SimpleInferer):
    def __call__(
        self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        ToDo: Add documentation
        """
        input_tensor = inputs["image"]  # Extracting only the image data
        meta = input_tensor.meta
        
        input_tensor_onnx = input_tensor.permute(0, 4, 3, 2, 1)
        image_tensor_onnx = input_tensor_onnx[..., :1].numpy()
        guide_tensor_onnx = input_tensor_onnx[..., 1:].numpy()
        
        inputs_onnx = {
            "image": image_tensor_onnx,
            "guide": guide_tensor_onnx
            }
        # Output shape: (1, 32, 960, 320, 2)
        outputs_onnx = network.run(None, inputs_onnx)[0]
        
        # Transform the network prediction back to tensor
        outputs_tensor = torch.from_numpy(outputs_onnx)
        outputs_tensor = outputs_tensor.permute(0, 4, 3, 2, 1).to(dtype=torch.float32)
        outputs = MetaTensor(outputs_tensor, meta=meta) 
        return outputs
