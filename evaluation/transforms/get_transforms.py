from monai.data.folder_layout import FolderLayout
import logging
import os
import numpy as np

from monai.apps.deepedit.transforms import NormalizeLabelsInDatasetd
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Identityd,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    NormalizeIntensityd,
    KeepLargestConnectedComponentd,
    RepeatChanneld,
    Resized
)

from evaluation.transforms.custom_transforms import (
    AddEmptySignalChannels, 
    FindDiscrepancyRegions,
    AddGuidance,
    AddGuidanceSignal,
    ConnectedComponentAnalysisd
)


logger = logging.getLogger("evaluation_pipeline_logger")


SPACING_FOR_DINS = (1.7, 1.7, 7.8)
ORIENTATION_FOR_DINS = ("SRA")
SPACING_FOR_SW_FASTEDIT = (0.625, 0.625, 7.8)
ORIENTATION_FOR_SW_FASTEDIT = ("RSA")
SPACING_FOR_SIMPLECLICK = (-1, -1, -1)
ORIENTATION_FOR_SIMPLECLICK = ("RSA") 
TARGET_SIZE_FOR_SIMPLECLICK = (1024, 1024, -1)
SPACING_FOR_SAM2 = (-1, -1, -1)
ORIENTATION_FOR_SAM2 = ("RSA") 


def get_pre_transforms(args, device="cpu", input_keys=("image", "label", "connected_component_label")):    
    # Unified transforms
    transforms = [
        LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=input_keys),
        NormalizeLabelsInDatasetd(keys=["label", "connected_component_label"], label_names=args.labels),
        KeepLargestConnectedComponentd(keys="connected_component_label", num_components=args.num_lesions) if args.evaluation_mode != "global_corrective"
        else Identityd(keys="connected_component_label"),
        ConnectedComponentAnalysisd(keys="connected_component_label") if args.evaluation_mode != "global_corrective"
        else Identityd(keys="connected_component_label"),
    ]
    
    # Add model-specific transforms
    if args.network_type == "SW-FastEdit":
        spacing = SPACING_FOR_SW_FASTEDIT
        orientation = ORIENTATION_FOR_SW_FASTEDIT
        
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                Spacingd(keys='image', pixdim=spacing),
                Spacingd(keys=['label', 'connected_component_label'], pixdim=spacing, mode="nearest"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                DivisiblePadd(keys=input_keys, k=32, value=0),
                AddEmptySignalChannels(keys='image', device=device) 
            ]
        )
        
    elif args.network_type == "DINs":
        spacing = SPACING_FOR_DINS
        orientation = ORIENTATION_FOR_DINS
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                Spacingd(keys='image', pixdim=spacing),
                Spacingd(keys=['label', 'connected_component_label'], pixdim=spacing, mode="nearest"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
        
    elif args.network_type == "SimpleClick":
        spacing = SPACING_FOR_SIMPLECLICK
        orientation = ORIENTATION_FOR_SIMPLECLICK
        target_size = TARGET_SIZE_FOR_SIMPLECLICK
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=255.0, clip=True),
                Resized(keys=["image", "label", "connected_component_label"], spatial_size=target_size, mode=["area", "nearest", "nearest"]),
            ]
        )
        
    elif args.network_type == "SAM2":
        spacing = SPACING_FOR_SAM2
        orientation = ORIENTATION_FOR_SAM2
        transforms.extend(
            [
                Orientationd(keys=input_keys, axcodes=orientation),
                ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=255.0, clip=True),
            ]
        )
        
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}")    
    
    return Compose(transforms)


def get_interaction_pre_transforms(args, device="cpu"):
    transforms = [
        EnsureTyped(keys=["image", "connected_component_label_local", "pred_local"], device="cuda"),
        FindDiscrepancyRegions(keys="connected_component_label_local", pred_key="pred_local", discrepancy_key="discrepancy", device="cuda"),
        AddGuidance(keys="NA", pred_key="pred_local", label_key="connected_component_label_local", discrepancy_key="discrepancy", probability_key="probability", device="cuda", 
                    patch_size=(args.patch_size_discrepancy)),
        AddGuidanceSignal(keys="image", sigma=args.sigma, disks=(not args.no_disks), device="cuda",),
        EnsureTyped(keys=["image", "connected_component_label_local", "pred_local"], device="cpu"),
    ]
    return Compose(transforms)

def get_interaction_post_transforms(args, device="cpu"):
    # Expects "current_label" containing only a binary mask
        
    if ((args.network_type == "SW-FastEdit") or 
        (args.network_type == "DINs")):
        transforms = [
            Activationsd(keys="pred_local", softmax=True),
            AsDiscreted(keys="pred_local", argmax=True),
            EnsureTyped(keys="pred_local", device=device),
        ]
    elif ((args.network_type == "SimpleClick") or 
          (args.network_type == "SAM2")):
        transforms = [
            EnsureTyped(keys="pred_local", device=device)
        ]
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}") 
        
    return Compose(transforms)

def get_post_transforms(args, pre_transforms, device="cpu"):
    
    if args.save_predictions:
        predictions_output_dir = os.path.join(
            args.results_dir, "predictions", args.network_type, args.evaluation_mode,
            f"TestSet_{args.test_set_id}", f"fold_{args.fold}")
        
        if not os.path.exists(predictions_output_dir):
            os.makedirs(predictions_output_dir)
        
        nii_layout = FolderLayout(output_dir=predictions_output_dir, 
                                  postfix="", 
                                  extension=".nii.gz", 
                                  makedirs=False)
    else:
        nii_layout = None
    
    transforms = [
        Invertd(
            keys=["pred", "label"],
            orig_keys="image",
            nearest_interp=False,
            transform=pre_transforms,
        ),
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ) if args.save_predictions else Identityd(keys="pred")
    ]
    return Compose(transforms)
    