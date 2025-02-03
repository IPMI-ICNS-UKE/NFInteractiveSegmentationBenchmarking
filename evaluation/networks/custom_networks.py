from typing import Any
from time import time
from tqdm import tqdm
import logging
import os

import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from monai.inferers import SlidingWindowInferer
from monai.metrics import compute_dice # Needed for debugging of SimpleClick

from evaluation.utils.image_cache import ImageCache
# Importing SAM2 dependencies
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_video_predictor
# Importing SimpleClick from the forked repository folder
from model_code.SimpleClick_Neurofibroma.isegm.inference import utils
from model_code.SimpleClick_Neurofibroma.isegm.inference import clicker as clk
from model_code.SimpleClick_Neurofibroma.isegm.inference.predictors import get_predictor
# Importing STCN for SimpleClick 3D from the forked repository folder
from model_code.iSegFormer_Neurofibroma.maskprop.Med_STCN.model.eval_network import STCN
from model_code.iSegFormer_Neurofibroma.maskprop.Med_STCN.inference_core import InferenceCore
from model_code.iSegFormer_Neurofibroma.maskprop.Med_STCN.util.tensor_util import unpad

logger = logging.getLogger("evaluation_pipeline_logger")


class DINsNetwork(nn.Module):
    """
    Wrapper class for deploying a Deep Interactive Network (DINs) using an ONNX runtime inference session.

    This class loads a pre-trained ONNX model and provides a PyTorch-compatible forward method 
    to handle input processing and inference.

    Attributes:
        network (onnxruntime.InferenceSession): ONNX inference session for executing the DINs model.
        device (torch.device): The device (CPU/GPU) to which the output tensor is mapped.

    Args:
        model_path (str): Path to the pre-trained ONNX model file.
        providers (list): List of ONNX execution providers (e.g., `["CUDAExecutionProvider", "CPUExecutionProvider"]`).
        device (torch.device): PyTorch device where the output tensor will be stored (e.g., `torch.device("cuda")`).
    """
    def __init__(self, model_path, providers, device):
        """
        Initializes the DINs ONNX-based inference model.
        """
        super().__init__()
        self.network = ort.InferenceSession(
            model_path, 
            providers=providers
            )
        self.device = device
    
    def forward(self, x):
        """
        Performs forward pass using ONNX inference, handling input processing and tensor conversion.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height, width, depth, channels).
                              Expected format: (B, H, W, D, C), where:
                              - `B` = batch size
                              - `H` = height
                              - `W` = width
                              - `D` = depth
                              - `C` = channels (image + guidance)

        Returns:
            torch.Tensor: Output logits tensor of shape (B, height, width, depth, 2),
                          with values mapped to the specified `device`.

        Processing Steps:
            1. Reorders the input tensor dimensions to match ONNX expected format (B, C, D, H, W).
            2. Splits input into `image` and `guide` tensors.
            3. Converts the tensors to NumPy format for ONNX execution.
            4. Runs ONNX inference and retrieves the output logits.
            5. Converts the output back to a PyTorch tensor and reorders the dimensions.
            6. Transfers the output tensor to the specified device.
        """
        # Reorder input tensor to match ONNX input format (B, C, D, H, W)
        input_tensor_onnx = x.permute(0, 4, 2, 3, 1)
        
        # Extract image and guidance channels
        image_tensor_onnx = input_tensor_onnx[..., :1].cpu().numpy() # Extracts first channel (image)
        guide_tensor_onnx = input_tensor_onnx[..., 1:].cpu().numpy() # Extracts remaining channels (guidance)
        
        # Prepare ONNX input dictionary
        input_onnx = {
            "image": image_tensor_onnx,
            "guide": guide_tensor_onnx
            }
        
        # Perform ONNX inference
        output_onnx = self.network.run(None, input_onnx)[0]
        # Convert ONNX output to PyTorch tensor
        output_tensor = torch.from_numpy(output_onnx)
        # Reorder tensor back to (B, D, H, W, 2) and transfer to correct device
        output_tensor = output_tensor.permute(0, 4, 2, 3, 1).to(dtype=torch.float32)
        return output_tensor.to(self.device)


class SAM2Network(nn.Module):
    """
    Implements the SAM2 model for 3D interactive segmentation using video-based inference.

    This class manages caching, model initialization, and bidirectional propagation of segmentation 
    predictions in a 3D volume. It uses an inference state to maintain consistency across frames.

    Attributes:
        image_cache (ImageCache): Cache manager for storing image slices used in inference.
        model_path (str): Path to the pre-trained model file.
        config_name (str): Name of the configuration file.
        config_path (str): Path to the configuration file.
        device (str): Computation device (`cuda` or `cpu`).
        predictors (dict): Stores initialized predictors for different devices.
        inference_state (Any): Holds the inference state used for propagation.
    
    Args:
        model_path (str): Path to the ONNX or PyTorch model.
        config_path (str): Path to the model configuration file.
        cache_path (str): Directory path for caching input images.
        device (str): Computation device (`cuda` or `cpu`).
    """

    def __init__(self, model_path, config_path, cache_path, device):
        """
        Initializes the SAM2Network model for interactive segmentation.

        Args:
            model_path (str): Path to the model file.
            config_path (str): Path to the configuration YAML file.
            cache_path (str): Directory for caching images used in inference.
            device (str): Computation device (`cuda` or `cpu`).
        """
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
        """
        Runs 3D interactive segmentation with bidirectional propagation.

        Args:
            reset_state (bool): Whether to reset the inference state before processing.
            image_tensor (torch.Tensor): 3D image volume tensor of shape `(H, W, D)`.
            guidance (Dict[str, torch.Tensor]): Dictionary containing lesion/background interaction points.
            case_name (str): Unique identifier for the current case.

        Returns:
            np.ndarray: 3D binary segmentation mask of shape `(H, W, D)`.
        """
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
        
        # Prepare input image directory
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
        
        # Set expiry time for cached images
        self.image_cache.cached_dirs[video_dir] = time() + self.image_cache.cache_expiry_sec
        
        # Initialize inference state if required
        if reset_state:
            if self.inference_state:
                predictor.reset_state(self.inference_state)
            self.inference_state = predictor.init_state(video_path=video_dir)
        
        # Extract interaction points from guidance
        fps: dict[int, Any] = {}
        bps: dict[int, Any] = {}
        sids = set()
        
        for key in {"lesion", "background"}:
            point_tensor = np.array(guidance[key].cpu())
            logger.info(f"point tensor: {point_tensor}")
            if point_tensor.size == 0:
                continue # Skip if no interaction points
            else:
                for point_id in range(point_tensor.shape[1]):
                    point = point_tensor[:, point_id, :][0][1:] # Extract (x, y, slice_index)
                    logger.info(f"p: {point}")
                    sid = point[2]
                    
                    sids.add(sid)
                    kps = fps if key == "lesion" else bps
                    if kps.get(sid):
                        kps[sid].append([point[0], point[1]])
                    else:
                        kps[sid] = [[point[0], point[1]]]

        # Forward propagation
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
        
        # Backward propagation
        pred_backward = np.zeros(tuple(image_tensor.shape))
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state, reverse=True):
            logger.info(f"propagate: {out_frame_idx} - mask_logits: {out_mask_logits.shape}; obj_ids: {out_obj_ids}")
            pred_backward[:, :, out_frame_idx] = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        
        # Merge forward and backward propagation
        pred = np.logical_or(pred_forward, pred_backward) 
        return pred
    
    def forward(self, x):
        """
        Performs segmentation inference using SAM2 model.

        Args:
            x (Dict[str, Any]): Dictionary containing:
                - `"image"`: 3D image tensor.
                - `"guidance"`: Interaction guidance points.
                - `"case_name"`: Case identifier.
                - `"reset_state"`: Boolean flag to reset the inference state.

        Returns:
            torch.Tensor: Segmentation prediction of shape `(1, 1, H, W, D)`.
        """
        image = torch.squeeze(x["image"].cpu())[0]
        guidance = x["guidance"]
        case_name = x["case_name"]
        reset_state = x["reset_state"]
        output = self.run_3d(reset_state, image, guidance, case_name)
        output = torch.Tensor(output).unsqueeze(0).unsqueeze(0)
        return output.to(self.device)


class SimpleClick3DNetwork(nn.Module):
    """
    Implements a hybrid 3D interactive segmentation framework using SimpleClick for 2D segmentation 
    and STCN (Space-Time Correspondence Network) for 3D propagation.

    This model is designed to process volumetric medical images interactively, refining segmentation 
    based on user clicks and propagating corrections through adjacent slices.

    **NOTE:** This model is still under development and may not work properly.

    Attributes:
        simpleclick_path (str): Path to the pre-trained SimpleClick model.
        stcn_path (str): Path to the STCN propagation model.
        device (torch.device): Computational device (`cuda` or `cpu`).
        threshold (float): Threshold for binarizing SimpleClick predictions.
        memory_freq (int): Frequency of memory updates in STCN.
        include_last (bool): Whether to include the last frame in STCN propagation.
        top_k (int): Number of top features to retain in STCN.
        patch_size (Tuple[int, int, int]): Patch size for sliding window inference.
        overlap (float): Overlap percentage for patch-wise inference.
        sw_batch_size (int): Batch size for sliding window inference.

    Args:
        simpleclick_path (str): Path to the SimpleClick model file.
        stcn_path (str): Path to the STCN model file.
        device (str): Computational device (`cuda` or `cpu`).
        simple_click_threshold (float, optional): Threshold for binarizing SimpleClick outputs. Default is `0.5`.
        stcn_memory_freq (int, optional): Memory update frequency in STCN. Default is `1`.
        stcn_include_last (bool, optional): Whether to include the last frame in STCN propagation. Default is `True`.
        stcn_top_k (int, optional): Number of top features to retain in STCN. Default is `20`.
        stcn_patch_size (Tuple[int, int, int], optional): Patch size for sliding window inference. Default is `(480, 480, 32)`.
        stcn_overlap (float, optional): Overlap percentage for patch inference. Default is `0.25`.
        stcn_sw_batch_size (int, optional): Batch size for patch inference. Default is `1`.
    """
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
        """
        Initializes the `SimpleClick3DNetwork` model.
        """
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
        """
        Performs 2D interactive segmentation using SimpleClick.

        Args:
            image (torch.Tensor): 2D image tensor of shape `(1, 3, H, W)`.
            prev_prediction (torch.Tensor): Previous segmentation prediction `(1, 1, H, W)`.
            point_coords (list): List of user click coordinates.
            point_labels (list): List of corresponding labels for the points.
            ground_truth_slice (torch.Tensor): Ground truth segmentation slice `(1, 1, H, W)`.

        Returns:
            torch.Tensor: Updated segmentation mask `(1, 1, H, W)`.
        """
        # image - torch.Size([1, 3, 1024, 1024])
        # prev_prediction - torch.Size([1, 1, 1024, 1024])
        # point_coords - [[437, 718], ...]
        # [1, ...]
        
        # Convert tensors to numpy for processing
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [1024, 1024, 3]
        # image_np = np.moveaxis(image_np, 0, 1)
        # image_np = np.flip(image_np, axis=0)
        # image_np = np.flip(image_np, axis=1)
        
        # gt_np = ground_truth_slice.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) *255 # [1024, 1024, 1]
        # gt_np = np.moveaxis(gt_np, 0, 1)
        # gt_np = np.flip(gt_np, axis=0)
        # gt_np = np.flip(gt_np, axis=1)
        
        # Convert numpy to PIL image
        image_pil = Image.fromarray(image_np, mode="RGB")
        # image_pil0 = Image.fromarray(image_np[:, :, 0], mode="L")
        # image_pil1 = Image.fromarray(image_np[:, :, 1], mode="L")
        # image_pil2 = Image.fromarray(image_np[:, :, 2], mode="L")
        
        # gt_pil = Image.fromarray(gt_np[:, :, 0], mode="L")
        # p0, p1 = point_coords[0]
        # draw = ImageDraw.Draw(image_pil)
        # radius = 5  # Radius of the dot
        
        # draw.ellipse((p0 - radius, p1 - radius, p0 + radius, p1 + radius), fill="red")
        
        # draw_gt = ImageDraw.Draw(gt_pil)
        # radius = 5  # Radius of the dot
        # draw_gt.ellipse((p0 - radius, p1 - radius, p0 + radius, p1 + radius), fill="red")
        
        # Save intermediate images for debugging purposes
        # image_pil.save("output_image.png")
        # gt_pil.save("gt.png")
        # image_pil0.save("output_image0.png")
        # image_pil1.save("output_image1.png")
        # image_pil2.save("output_image2.png")
        
        # Load SimpleClick model        
        model = utils.load_is_model(self.simpleclick_path, self.device, eval_ritm=False, cpu_dist_maps=True)
        clicker = clk.Clicker()
        predictor = get_predictor(model, device=self.device, brs_mode='NoBRS')
        
        # Add user interactions to the predictor
        for point, label in zip(point_coords, point_labels):
            click = clk.Click(is_positive=label, coords=(point[0], point[1]))
            clicker.add_click(click)
        
        predictor.set_input_image(image_pil)
        
        # Perform inference
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
        """
        Propagates 2D segmentation across slices using STCN.

        Args:
            input_concat (torch.Tensor): Concatenated image and segmentation `(1, 4, H, W, D)`.

        Returns:
            torch.Tensor: 3D segmentation prediction `(1, 1, H, W, D)`.
        """
        # This method was not debugged yet
        # input_concat [1, 3+1, 1024, 1024, 32]
        image_patch = input_concat[:, :3, :, :, :]
        mask_patch = input_concat[:, 3:, :, :, :]
        
        image_patch_reshaped = image_patch.permute(0, 4, 1, 2, 3)
        mask_patch_reshaped = mask_patch.permute(0, 4, 1, 2, 3)
        logger.info(f"Before prediction: {torch.sum(mask_patch_reshaped)}")
        # Image: [1, T, 3, 480, 480]; Mask: [N, T, 1, 480, 480]
        _, num_frames, _, height, width = mask_patch_reshaped.shape
        
        # Perform STCN-based segmentation propagation
        model = STCN().cuda().eval()
        # Performs input mapping such that stage 0 model can be loaded
        prop_saved = torch.load(self.stcn_path)
        
        for k in list(prop_saved.keys()):
            if k == 'value_encoder.conv1.weight':
                if prop_saved[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                    prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
        model.load_state_dict(prop_saved)
        
        # Find the best starting frame
        max_area, max_area_idx = -1, num_frames // 2
        for i in range(num_frames):
            area = torch.count_nonzero(mask_patch_reshaped[:,i])
            # print(area)
            if area > max_area:
                max_area = area
                max_area_idx = i
        # logger.info(f"Before prediction: {torch.sum(mask_patch_reshaped[:, max_area_idx])}")
        
        # Perform STCN-based segmentation propagation
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
        
        return out_masks
    
    def forward(self, x):
        """
        Performs full 3D interactive segmentation using SimpleClick for 2D slices and STCN for 3D propagation.

        Args:
            x (Dict[str, Any]): Dictionary containing:
                - `"image"`: 3D image tensor `(1, 1, H, W, D)`.
                - `"previous_prediction"`: Previous segmentation `(1, 1, H, W, D)`.
                - `"guidance"`: Dictionary of lesion/background user clicks.
                - `"gt"`: Ground truth segmentation `(1, 1, H, W, D)`.

        Returns:
            torch.Tensor: 3D segmentation prediction `(1, 1, H, W, D)`.
        """
        torch.backends.cudnn.deterministic = True        
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
        # Form input for 3D propagation
        input_concat = torch.cat([image, previous_prediction], dim=1)

        # Patch-wise processing using SlidingWindowInferer
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
