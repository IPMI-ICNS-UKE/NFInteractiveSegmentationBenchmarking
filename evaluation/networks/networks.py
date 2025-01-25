from monai.networks.nets.dynunet import DynUNet
import logging
import os
import torch
from typing import Iterable
import onnxruntime as ort
import torch.nn as nn
from monai.data import MetaTensor
import numpy as np

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
        network.to(device)
        
    elif args.network_type == "DINs":
        # The original DINs model was trained with and Tensorflow 2.8 version.
        # For re-usability the DINs model was exported as ONNX 
        # and launched as an ONNX runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        network = DINsNetwork(model_path, providers, device)
        
    elif args.network_type == "SimpleClick":
        raise NotImplementedError(f"Network type is not implemented yet: {args.network_type}")
    elif args.network_type == "SAM2":
        raise NotImplementedError(f"Network type is not implemented yet: {args.network_type}")
    else:
        raise ValueError(f"Unsupported network: {args.network_type}")

    logger.info(f"Selected network: {args.network_type}")

    return network


class DINsNetwork(nn.Module):
    def __init__(self, model_path, providers, device):
        super().__init__()
        self.network = ort.InferenceSession(
            model_path, 
            providers=providers
            )
        self.device = device
    
    def forward(self, x):                
        input_tensor_onnx = x.permute(0, 4, 2, 3, 1)
        image_tensor_onnx = input_tensor_onnx[..., :1].cpu().numpy()
        guide_tensor_onnx = input_tensor_onnx[..., 1:].cpu().numpy()
        input_onnx = {
            "image": image_tensor_onnx,
            "guide": guide_tensor_onnx
            }
        
        output_onnx = self.network.run(None, input_onnx)[0]
        # Logits with shape (1, 10, 512, 160, 2)
        output_tensor = torch.from_numpy(output_onnx)
        output_tensor = output_tensor.permute(0, 4, 2, 3, 1).to(dtype=torch.float32)
        return output_tensor.to(self.device)
    