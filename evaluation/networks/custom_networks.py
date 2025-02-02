import logging
import os
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from typing import Any
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from evaluation.utils.image_cache import ImageCache

from model_code.SimpleClick_Neurofibroma.isegm.inference import utils
from model_code.SimpleClick_Neurofibroma.isegm.inference import clicker as clk
from model_code.SimpleClick_Neurofibroma.isegm.inference.predictors import get_predictor

from model_code.iSegFormer_Neurofibroma.maskprop.Med_STCN.model.eval_network import STCN
from model_code.iSegFormer_Neurofibroma.maskprop.Med_STCN.inference_core import InferenceCore
from model_code.iSegFormer_Neurofibroma.maskprop.Med_STCN.util.tensor_util import unpad

from monai.inferers import SlidingWindowInferer

import torch
import numpy as np
from PIL import Image, ImageDraw

from monai.metrics import compute_dice


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


class SimpleClick3DNetwork(nn.Module):
    def __init__(self, 
                 simpleclick_path, 
                 stcn_path, 
                 device,
                 simple_click_threshold=0.5, 
                 stcn_memory_freq=1, 
                 stcn_include_last=True, 
                 stcn_top_k=20,
                 stcn_patch_size=(480, 480, 32),
                 stcn_overlap=0.25,
                 stcn_sw_batch_size=1
                 ):
        super().__init__()
        logger.info("The SimpleClick model is still under development and does not work properly yet!!!")
        self.simpleclick_path = simpleclick_path
        self.stcn_path = stcn_path
        self.device = device
        self.threshold = simple_click_threshold
        self.memory_freq = stcn_memory_freq
        self.include_last = stcn_include_last
        self.top_k = stcn_top_k
        self.patch_size = stcn_patch_size
        self.overlap = stcn_overlap
        self.sw_batch_size = stcn_sw_batch_size
        
    def run_simple_click_2d(self, image, prev_prediction, point_coords, point_labels, ground_truth_slice):
        # image - torch.Size([1, 3, 1024, 1024])
        # prev_prediction - torch.Size([1, 1, 1024, 1024])
        # point_coords - [[437, 718], ...]
        # [1, ...]
        
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [1024, 1024, 3]
        # image_np = np.moveaxis(image_np, 0, 1)
        # image_np = np.flip(image_np, axis=0)
        # image_np = np.flip(image_np, axis=1)
        
        gt_np = ground_truth_slice.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) *255 # [1024, 1024, 1]
        # gt_np = np.moveaxis(gt_np, 0, 1)
        # gt_np = np.flip(gt_np, axis=0)
        # gt_np = np.flip(gt_np, axis=1)
        
        image_pil = Image.fromarray(image_np, mode="RGB")
        image_pil0 = Image.fromarray(image_np[:, :, 0], mode="L")
        image_pil1 = Image.fromarray(image_np[:, :, 1], mode="L")
        image_pil2 = Image.fromarray(image_np[:, :, 2], mode="L")
        
        gt_pil = Image.fromarray(gt_np[:, :, 0], mode="L")
        p0, p1 = point_coords[0]
        draw = ImageDraw.Draw(image_pil)
        radius = 5  # Radius of the dot
        
        draw.ellipse((p0 - radius, p1 - radius, p0 + radius, p1 + radius), fill="red")
        
        draw_gt = ImageDraw.Draw(gt_pil)
        radius = 5  # Radius of the dot
        draw_gt.ellipse((p0 - radius, p1 - radius, p0 + radius, p1 + radius), fill="red")
        
        # Save intermediate images for debugging purposes
        # image_pil.save("output_image.png")
        # gt_pil.save("gt.png")
        # image_pil0.save("output_image0.png")
        # image_pil1.save("output_image1.png")
        # image_pil2.save("output_image2.png")
                
        model = utils.load_is_model(self.simpleclick_path, self.device, eval_ritm=False, cpu_dist_maps=True)
        clicker = clk.Clicker()
        predictor = get_predictor(model, device=self.device, brs_mode='NoBRS')
        
        for point, label in zip(point_coords, point_labels):
            click = clk.Click(is_positive=label, coords=(point[0], point[1]))
            clicker.add_click(click)
        
        predictor.set_input_image(image_pil)
        
         # Pass the data to the model
        prediction = predictor.get_prediction(clicker, prev_mask=prev_prediction)
        prediction = (prediction >= self.threshold).astype(np.uint8)
        
        prediction = torch.tensor(prediction, dtype=prev_prediction.dtype, device=prev_prediction.device)
        prediction = prediction.unsqueeze(0).unsqueeze(0)
        
        # Debugging
        # print("0"*100)
        # print(torch.sum(prediction), torch.sum(ground_truth_slice))
        # print(compute_dice(y_pred=prediction,
        #                    y=ground_truth_slice,
        #                    include_background=False
        #                    ))

        logger.info(f"Input: {prev_prediction.shape}, output: {prediction.shape}, sum: {torch.sum(prediction)}")
        # Ensure that the prediction has the same type and shape as input
        return prediction
        
    
    def run_stcn_propagation_3d(self, input_concat):
        # This method was not debugged yet
        # input_concat [1, 3+1, 1024, 1024, 32]
        image_patch = input_concat[:, :3, :, :, :]
        mask_patch = input_concat[:, 3:, :, :, :]
        
        image_patch_reshaped = image_patch.permute(0, 4, 1, 2, 3)
        mask_patch_reshaped = mask_patch.permute(0, 4, 1, 2, 3)
        logger.info(f"Before prediction: {torch.sum(mask_patch_reshaped)}")
        # Image: [1, T, 3, 480, 480]; Mask: [N, T, 1, 480, 480]
        _, num_frames, _, height, width = mask_patch_reshaped.shape
        
        model = STCN().cuda().eval()
        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load(self.stcn_path)
        
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        model.load_state_dict(prop_saved)
        
        # find the best starting frame
        max_area, max_area_idx = -1, num_frames // 2
        for i in range(num_frames):
            area = torch.count_nonzero(mask_patch_reshaped[:,i])
            # print(area)
            if area > max_area:
                max_area = area
                max_area_idx = i
        # logger.info(f"Before prediction: {torch.sum(mask_patch_reshaped[:, max_area_idx])}")
        
        processor = InferenceCore(model, 
                                  image_patch_reshaped, 
                                  num_objects=1, 
                                  top_k=self.top_k,
                                  mem_every=self.memory_freq, 
                                  include_last=self.include_last)
        processor.interact(mask_patch_reshaped[:, max_area_idx], max_area_idx)

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, height, width), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, (height, width), mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
        # Mask: [N, T, 1, 480, 480]
        out_masks = out_masks.unsqueeze(0) # (1, 16, 1, 480, 480)
        out_masks = out_masks.permute(0, 2, 3, 4, 1) # (1, 1, 480, 480, 16)
        
        out_masks = torch.tensor(out_masks, dtype=input_concat.dtype, device=input_concat.device)
        logger.info(f"SUM prediction: {torch.sum(out_masks)}")
        logger.info(f"STCN prediction: {out_masks.shape}")
        
        # Ensure that the prediction has the same type and shape as input
        return out_masks
    
    def forward(self, x):
        torch.backends.cudnn.deterministic = True
        #### CHECK THE POINT ORIENTATION
        
        # Check whether squeeze is needed
        
        image = x["image"][:, :1, :, :, :] # torch.Size([1, 3, 1024, 1024, 32])
        image = image.repeat(1, 3, 1, 1, 1)
        previous_prediction = x["previous_prediction"] # torch.Size([1, 1, 1024, 1024, 32])
        guidance = x["guidance"] # {'lesion': tensor([[[  0, 718, 437,   9]]], device='cuda:0', dtype=torch.int32), 'background': tensor([], device='cuda:0', size=(1, 0), dtype=torch.int32)}
        ground_truth = x["gt"]
        
        logger.info(f"Got image: {image.shape}, prev_pred: {previous_prediction.shape}, guidance: {guidance}")
        
        # 2D interactive segmentation wit SimpleClick
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        
        # Get points
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
        logger.info(f"Formed FPs: {fps}, and BPs: {bps}")
        
        # Inference
        for sid in sorted(sids):
            
            fp = fps.get(sid, [])
            bp = bps.get(sid, [])
            
            point_coords = fp + bp
            point_coords = [[p[1], p[0]] for p in point_coords]  # Flip x,y => y,x Check whether it is correct
            # point_coords = [[p[0], p[1]] for p in point_coords]  # Flip x,y => y,x Check whether it is correct
            point_labels = [1] * len(fp) + [0] * len(bp)
            logger.info(f"{sid} - Coords: {point_coords}; Labels: {point_labels}") # 9 - Coords: [[437, 718], ...]; Labels: [1, ...]
            
            logger.info(f"Inferencing slice with SimpelClick: {sid}")
            
            image_slice = image[:, :, :, :, sid]  # torch.Size([1, 3, 1024, 1024])
            ground_truth_slice = ground_truth[:, :, :, :, sid]
            previous_prediction_slice = previous_prediction[:, :, :, :, sid] # torch.Size([1, 1, 1024, 1024])
            
            updated_slice = self.run_simple_click_2d(image_slice, 
                                                     previous_prediction_slice,
                                                     point_coords,
                                                     point_labels,
                                                     ground_truth_slice
                                                     )
            # Insert updated slice back to 3D prediction
            # Debugging the 2D output prediction of the SimpleClick model
            # print("BEFORE INSERT: ", torch.sum(previous_prediction))
            previous_prediction[:, :, :, :, sid] = updated_slice # [1, 3+1, 1024, 1024, 32]
            # print("After INSERT: ", torch.sum(previous_prediction))
        logger.info(f"Finished SimpleClick inference")
        
        # 3D propagation
        # Step 5: Form input for 3D propagation
        input_concat = torch.cat([image, previous_prediction], dim=1)

        # Step 6: Patch-wise processing using SlidingWindowInferer
        inferer = SlidingWindowInferer(
            roi_size=self.patch_size,
            sw_batch_size=self.sw_batch_size,  # Process one patch at a time
            overlap=self.overlap,
            mode="gaussian"
        )
                    
        # Run inference 
        output_prediction = inferer(
            inputs=input_concat,
            network=self.run_stcn_propagation_3d  # Custom function for patch processing
        )

        # Retrieve the final 3D prediction
        logger.info(f"Prediction: {output_prediction.shape}")
        return output_prediction.to(self.device)
