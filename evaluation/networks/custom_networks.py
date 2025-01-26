import logging
import os
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
from time import time
from typing import Any
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from evaluation.utils.image_cache import ImageCache

logger = logging.getLogger("evaluation_pipeline_logger")


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


class SAM2Network(nn.Module):
    def __init__(self, model_path, config_path, cache_path, device):
        super().__init__()
        self.image_cache = ImageCache(cache_path)
        self.image_cache.monitor()
        model_dir = os.path.dirname(model_path)
        self.model_path = model_path
        self.config_name = os.path.basename(config_path)
        self.config_path = config_path 
        self.device = device
        
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=model_dir)
        
        self.predictors = {}
        self.inference_state = None
    
    def run_3d(self, reset_state, image_tensor, guidance, case_name):
        predictor = self.predictors.get(self.device)
        
        if predictor is None:
            logger.info(f"Using Device: {self.device}")
            device_t = torch.device(self.device)
            if device_t.type == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            predictor = build_sam2_video_predictor(self.config_name, self.model_path, device=self.device)
            self.predictors[self.device] = predictor
        
        video_dir = os.path.join(
            self.image_cache.cache_path, case_name
        ) 
        
        logger.info(f"Image: {image_tensor.shape}")
        
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            for slice_idx in tqdm(range(image_tensor.shape[-1])):
                slice_np = image_tensor[:, :, slice_idx].numpy()
                slice_file = os.path.join(video_dir, f"{str(slice_idx).zfill(5)}.jpg")
                Image.fromarray(slice_np).convert("RGB").save(slice_file)
            logger.info(f"Image (Flattened): {image_tensor.shape[-1]} slices; {video_dir}")
        
        # Set Expiry Time
        self.image_cache.cached_dirs[video_dir] = time() + self.image_cache.cache_expiry_sec
        
        if reset_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state)
            self.inference_state = predictor.init_state(video_path=video_dir)
        
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        
        for key in {"lesion", "background"}:
            # ToDo: Need to double-check the order of guidance points
            point_tensor = np.array(guidance[key].cpu())
            logger.info(f"point tensor: {point_tensor}")
            if point_tensor.size == 0:
                continue # No interaction points
            else:
                for point_id in range(point_tensor.shape[1]):
                    point = point_tensor[:, point_id, :][0][1:]
                    logger.info(f"p: {point}")
                    sid = point[2]
                    
                    sids.add(sid)
                    kps = fps if key == "lesion" else bps
                    if kps.get(sid):
                        kps[sid].append([point[0], point[1]])
                    else:
                        kps[sid] = [[point[0], point[1]]]

        # Forward inference
        pred_forward = np.zeros(tuple(image_tensor.shape))
        for sid in sorted(sids):
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])
            
            point_coords = fp + bp
            point_coords = [[p[1], p[0]] for p in point_coords]  # Flip x,y => y,x
            point_labels = [1] * len(fp) + [0] * len(bp)
            logger.info(f"{sid} - Coords: {point_coords}; Labels: {point_labels}")
            
            o_frame_ids, o_obj_ids, o_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=sid,
                obj_id=0,
                points=np.array(point_coords) if point_coords else None,
                labels=np.array(point_labels) if point_labels else None,
                box=None,
            )
            pred_forward[:, :, sid] = (o_mask_logits[0][0] > 0.0).cpu().numpy()
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_forward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        # Backward inference
        pred_backward = np.zeros(tuple(image_tensor.shape))

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, reverse=True):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_backward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        pred = np.logical_or(pred_forward, pred_backward) # Merge forward and backward propagation
            
        return pred
    
    def forward(self, x):
        # Reset state
        image = torch.squeeze(x["image"].cpu())[0]
        guidance = x["guidance"]
        case_name = x["case_name"]
        reset_state = x["reset_state"]
        output = self.run_3d(reset_state, image, guidance, case_name)
        output = torch.Tensor(output).unsqueeze(0).unsqueeze(0)
        return output.to(self.device)
