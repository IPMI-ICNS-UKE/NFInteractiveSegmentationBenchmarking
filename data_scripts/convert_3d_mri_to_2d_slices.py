import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from PIL import Image
from tqdm import tqdm


# Low-level processing
def reorient_to_ras(image):
    """
    Reorient the given image to RAS orientation using SimpleITK.
    
    Parameters:
    image (SimpleITK.Image): The input image.
    
    Returns:
    SimpleITK.Image: The reoriented image.
    """
    ras_orient_filter = sitk.DICOMOrientImageFilter()
    ras_orient_filter.SetDesiredCoordinateOrientation("RAS")
    return ras_orient_filter.Execute(image)


def preprocess_image(image_data, method="percentile"):
    """
    Preprocess the image using intensity normalization.
    
    Parameters:
    image_data (numpy.ndarray): The input image data.
    method (str): The normalization method ('percentile' or 'mean').
    
    Returns:
    numpy.ndarray: The preprocessed image data.
    """
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
    """
    Generates an instance mask from a binary segmentation mask by labeling connected components.

    Args:
        segmentation (np.ndarray): Binary segmentation mask (2D or 3D).
        min_size (int, optional): Minimum size threshold to retain components. Defaults to 500.
        remove_small (bool, optional): If True, removes components smaller than `min_size`. Defaults to True.
        limit_to_255 (bool, optional): If True, limits the number of components to the largest 255. Defaults to False.

    Returns:
        np.ndarray: Instance-labeled mask where each connected component has a unique ID.
    """
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
    """
    Saves 2D slices from a 3D numpy array into image files.

    Args:
        data (np.ndarray): A 3D numpy array (D, H, W) representing image slices.
        output_folder (str): The directory where slices will be saved.
        file_format (str, optional): Image format to save ('png' or 'jpg'). Defaults to 'png'.
        prefix (str, optional): Prefix for saved filenames. Defaults to "slice_".

    Raises:
        ValueError: If an unsupported file format is provided.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, slice_ in enumerate(data):
        filename = os.path.join(output_folder, f"{prefix}{i:05d}.{file_format}")
        if file_format == "jpg":
            image = Image.fromarray(slice_.astype(np.uint8))
            image.save(filename)
        elif file_format == "png":
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
    """
    Processes a single case by loading, preprocessing, and saving 2D slices from 3D medical images.

    Args:
        case_path (tuple): Tuple containing paths to (image, mask).
        output_root (str): Root directory for saving processed slices.
        preprocess_method (str): Preprocessing method for the image.
        remove_small_tumors (bool): If True, removes tumors smaller than `min_tumor_size`.
        limit_to_255 (bool): If True, limits instance mask labels to a maximum of 255 components.
        use_instance_masks (bool): If True, generates instance-wise segmentation masks.
        only_labels (bool): If True, processes only the labels without loading images.
        min_tumor_size (int, optional): Minimum tumor size threshold for filtering. Defaults to 500.

    Raises:
        FileNotFoundError: If the image or mask path is missing.
    """
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
    """
    Processes multiple datasets by iterating through image and mask pairs.

    Args:
        input_root (str): Root directory containing dataset folders.
        output_root (str): Directory to save processed slices.
        dataset_folders (list): List of dataset folder names within `input_root`.
        preprocess_method (str): Preprocessing method for images.
        remove_small_tumors (bool): If True, removes small tumor regions.
        limit_to_255 (bool): If True, limits the number of unique instance labels to 255.
        use_instance_mask (bool): If True, processes masks as instance segmentation masks.
        only_labels (bool): If True, processes only label masks without images.

    Raises:
        FileNotFoundError: If an expected dataset folder is missing.
    """
    for dataset in dataset_folders:
        print(f"Processing dataset: {dataset}")
        image_folder = os.path.join(input_root, dataset)
        mask_folder = image_folder.replace("images", "labels")
        image_files = sorted(os.listdir(image_folder))
        mask_files = sorted(os.listdir(mask_folder))

        for image_file, mask_file in tqdm(zip(image_files, mask_files)):
            image_path = os.path.join(image_folder, image_file)
            mask_path = os.path.join(mask_folder, mask_file)
            
            try:
                process_case(
                    (image_path, mask_path),
                    output_root,
                    preprocess_method,
                    remove_small_tumors,
                    limit_to_255,
                    use_instance_mask,
                    only_labels,
                )
            except Exception as e:
                print(f"Error processing case {image_file} / {mask_file}: {e}")
                

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
