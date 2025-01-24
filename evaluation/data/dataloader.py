import glob
import logging
import os
from typing import List
from monai.data import DataLoader, Dataset


logger = logging.getLogger("evaluation_pipeline_logger")


def get_evaluation_datalist(args) -> List:
    logger.info(f"Getting data list from: {os.path.join(args.input_dir, f'imagesTs_{args.test_set_id}')}")
    
    # Get all available images and labels
    image_files = sorted(glob.glob(os.path.join(args.input_dir, f"imagesTs_{args.test_set_id}", "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(args.input_dir, f"labelsTs_{args.test_set_id}", "*.nii.gz")))
    
    # Build the train and validation datasets
    datalist = [
        {"image": image_name, "label": label_name, "connected_component_label": label_name}
        for image_name, label_name in zip(image_files, label_files)
    ]
    logger.info(f"Number of evaluation cases: {len(datalist)}")
    
    if args.limit:
        logger.info(f"Limiting the evaluation cases to: {args.limit}")
        datalist = datalist[0 : args.limit]
        
    return datalist

def get_evaluation_data_loader(args, pre_transforms):
    datalist = get_evaluation_datalist(args)
    
    if not len(datalist):
        raise ValueError("No valid data found..")

    dataset = Dataset(datalist, pre_transforms)
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    return data_loader
