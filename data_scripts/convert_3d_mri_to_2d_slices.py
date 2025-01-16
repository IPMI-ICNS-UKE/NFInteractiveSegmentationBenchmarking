import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from PIL import Image
import imageio
from tqdm import tqdm


# Low-level processing
def reorient_to_ras(image):
    """
    Reorient the given image to RAS orientation using SimpleITK.
    """
    ras_orient_filter = sitk.DICOMOrientImageFilter()
    ras_orient_filter.SetDesiredCoordinateOrientation("RAS")
    return ras_orient_filter.Execute(image)


def preprocess_image(image_data, method="percentile"):
    if method == "percentile":
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
    else:
        mean = np.mean(image_data[image_data > 0])
        std = np.std(image_data[image_data > 0])
        image_data_pre = (image_data - mean) / std
        image_data_pre[image_data == 0] = 0
    return image_data_pre


def create_instance_mask(segmentation, min_size=500, remove_small=True, limit_to_255=False):
    labeled_mask = label(segmentation > 0, connectivity=3)
    unique, counts = np.unique(labeled_mask, return_counts=True)
    
    if remove_small:
        # Exclude the background (index 0)
        components = [(instance_id, size) for instance_id, size in zip(unique[1:], counts[1:]) if size >= min_size]
    else:
        # Include all components
        components = [(instance_id, size) for instance_id, size in zip(unique[1:], counts[1:])]
    
    # Sort components by size in descending order
    components = sorted(components, key=lambda x: x[1], reverse=True)
    
    # If limit_to_255 is True, keep only the largest 255 components
    if limit_to_255:
        # Limit to the 255 largest components
        components = components[:255]
        # Create the output mask with values in the range [0, 255]
        instance_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
        for new_id, (instance_id, _) in enumerate(components, start=1):  # Start IDs from 1
            instance_mask[labeled_mask == instance_id] = new_id
    else:
        # Create the output mask without limiting to 255 components
        instance_mask = np.zeros_like(labeled_mask)
        for instance_id, size in components:
            instance_mask[labeled_mask == instance_id] = instance_id
    
    return instance_mask


def save_slices(data, output_folder, file_format, prefix):
    os.makedirs(output_folder, exist_ok=True)
    for i, slice_ in enumerate(data):
        filename = os.path.join(output_folder, f"{prefix}{i:05d}.{file_format}")
        if file_format == "jpg":
            image = Image.fromarray(slice_.astype(np.uint8))
            image.save(filename)
        elif file_format == "png":
            # print(np.unique(slice_))
            # imageio.imwrite(filename, slice_)
            if np.max(slice_) > 255:
                slice_ = slice_.astype(np.uint16)
            else:
                slice_ = slice_.astype(np.uint8)
            image = Image.fromarray(slice_)
            image.save(filename)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


# Case-level processing
def process_case(case_path, output_root, preprocess_method, remove_small_tumors, limit_to_255,
                 use_instance_masks, only_labels, min_tumor_size=500):
    image_path, mask_path = case_path
    if not only_labels:
        # Read and reorient image and mask to RAS
        image_sitk = sitk.ReadImage(image_path)
        image_sitk = reorient_to_ras(image_sitk)
        image = sitk.GetArrayFromImage(image_sitk)
        
        # Preprocess image
        preprocessed_image = preprocess_image(image, method=preprocess_method)
    
    mask_sitk = sitk.ReadImage(mask_path)
    mask_sitk = reorient_to_ras(mask_sitk)
    mask = sitk.GetArrayFromImage(mask_sitk)

    
    # Generate instance mask
    if use_instance_masks:
        instance_mask = create_instance_mask(mask, min_size=min_tumor_size, remove_small=remove_small_tumors, limit_to_255=limit_to_255)
    else:
        instance_mask = mask > 0
        # instance_mask = instance_mask.astype(np.uint8) * 255
        instance_mask = instance_mask.astype(np.uint8)

    # Slice along the shortest axis
    axis = np.argmin(mask.shape)
    instance_slices = np.moveaxis(instance_mask, axis, 0)
    instance_slices = np.flip(instance_slices, axis=1)
    instance_slices = np.flip(instance_slices, axis=2)
    
    if not only_labels:
        image_slices = np.moveaxis(preprocessed_image, axis, 0)
        image_slices = np.flip(image_slices, axis=1)
        image_slices = np.flip(image_slices, axis=2)
    
    
    # Define folder structure
    case_name = os.path.splitext(os.path.basename(image_path))[0].split(".")[0]
    dataset_folder_image = os.path.basename(os.path.dirname(image_path))
    
    if use_instance_masks:
        dataset_folder_mask = os.path.basename(os.path.dirname(mask_path)) + "_instance"
    else:
        dataset_folder_mask = os.path.basename(os.path.dirname(mask_path)) + "_semantic"
    pics_folder_image = os.path.join(output_root, dataset_folder_image, case_name)
    
    pics_folder_instance = os.path.join(output_root, dataset_folder_mask, case_name)
    
    if not only_labels:
        # Save image slices
        save_slices(image_slices, pics_folder_image, "jpg", prefix="")

    # Save instance mask slices
    save_slices(instance_slices, pics_folder_instance, "png", prefix="")


# Dataset-level processing
def process_dataset(input_root, output_root, dataset_folders, preprocess_method, 
                    remove_small_tumors, limit_to_255, use_instance_mask, only_labels):
    for dataset in dataset_folders:
        print(f"Processing dataset: {dataset}")
        image_folder = os.path.join(input_root, dataset)
        mask_folder = image_folder.replace("images", "labels")
        image_files = sorted(os.listdir(image_folder))
        mask_files = sorted(os.listdir(mask_folder))

        for image_file, mask_file in tqdm(zip(image_files, mask_files)):
            image_path = os.path.join(image_folder, image_file)
            mask_path = os.path.join(mask_folder, mask_file)
            process_case(
                (image_path, mask_path),
                output_root,
                preprocess_method,
                remove_small_tumors,
                limit_to_255,
                use_instance_mask,
                only_labels,
            )


if __name__ == "__main__":
    # Parameters
    input_root = "../data/raw/"
    output_root = "../data/processed/"
    preprocess_method = "percentile"
    use_instance_mask = True
    remove_small_tumors = False
    limit_to_255 = True
    only_labels = False
    
    dataset_folders =  os.listdir(input_root)
    
    process_dataset(input_root, output_root, dataset_folders, preprocess_method, remove_small_tumors, limit_to_255, use_instance_mask, only_labels)
