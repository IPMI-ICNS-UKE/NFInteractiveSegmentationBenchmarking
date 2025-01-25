# Copyright 2019-2020 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import pickle
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tqdm

ROOT = Path(__file__).parents[1]
DATA_ROOT = ROOT / "data/NF"


def read_nii(file_name, out_dtype=np.int16, special=False, only_header=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    if only_header:
        return vh
    affine = vh.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    data = nib_vol.get_fdata().astype(out_dtype).transpose(*trans[::-1])
    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def write_nii(data, header, out_path, out_dtype=np.int16, special=False, affine=None):
    if header is not None:
        affine = header.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    trans_bk = [np.argwhere(np.array(trans[::-1]) == i)[0][0] for i in range(3)]

    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:  # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:  # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:  # Increase z from Interior to Superior
        data = np.flip(data, axis=0)

    out_image = np.transpose(data, trans_bk).astype(out_dtype)
    if header is None and affine is not None:
        out = nib.Nifti1Image(out_image, affine=affine)
    else:
        out = nib.Nifti1Image(out_image, affine=None, header=header)
    nib.save(out, str(out_path))


def load_data(logger):
    data_dir = DATA_ROOT / "nii_NF"
    path_list = list(data_dir.glob("volume*"))

    logger.info(' ' * 11 + f"==> Loading data ({len(path_list)} examples) ...")
    cache_path = DATA_ROOT / "cache.pkl.gz"
    if cache_path.exists():
        logger.info(' ' * 11 + f"==> Loading data cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            _data_cache = pickle.loads(data)
        logger.info(' ' * 11 + "==> Finished!")
        return _data_cache

    _data_cache = {}
    for path in tqdm.tqdm(path_list):
        pid = path.name.split(".")[0].split("-")[-1]
        header, volume = read_nii(path)
        la_path = path.parent / path.name.replace("volume", "segmentation")
        _, label = read_nii(la_path)
        assert volume.shape == label.shape, f"{volume.shape} vs {label.shape}"
        _data_cache[int(pid)] = {"im_path": path.absolute(),
                                 "la_path": la_path.absolute(),
                                 "img": volume,
                                 "lab": label.astype(np.uint8),
                                 "pos": np.stack(np.where(label > 0), axis=1),
                                 "meta": header,
                                 "lab_rng": np.unique(label)}
    with cache_path.open("wb") as f:
        logger.info(' ' * 11 + f"==> Saving data cache to {cache_path}")
        cache_s = pickle.dumps(_data_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    logger.info(' ' * 11 + "==> Finished!")
    return _data_cache


def load_split(set_key, test_fold, fold_path):
    if set_key in ["train", "eval_online", "eval"]:
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        val_split = folds.loc[folds.split == test_fold]
        if set_key != "train":
            return val_split
        train_folds = list(range(5))
        train_folds.remove(test_fold)
        train_split = folds.loc[folds.split.isin(train_folds)]
        return train_split
    elif set_key == "test":
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        test_split = folds.loc[folds.split == 0]
        return test_split
    else:
        raise ValueError(f"`set_key` supports [train|eval_online|eval|test|extra], got {set_key}")


def filter_tiny_nf(mask):
    struct2 = ndi.generate_binary_structure(2, 1)
    for i in range(mask.shape[0]):
        res, n_obj = ndi.label(mask[i], struct2)
        size = np.bincount(res.flat)
        for j in np.where(size <= 2)[0]:
            mask[i][res == j] = 0

    struct3 = ndi.generate_binary_structure(3, 2)
    res, n_obj = ndi.label(mask, struct3)
    size = np.bincount(res.flat)
    for i in np.where(size <= 5)[0]:
        mask[res == i] = 0
    return mask


def slim_labels(data, logger, opt=None):
    if not opt.uke_dataset:
        # Use default data root
        slim_labels_path = DATA_ROOT / "slim_labels.pkl.gz"
    else:
        slim_labels_path = opt.data_root / "slim_labels.pkl.gz"
    if slim_labels_path.exists():
        logger.info(' ' * 11 + f"==> Loading slimmed label cache from {slim_labels_path}")
        with slim_labels_path.open("rb") as f:
            new_labels = pickle.loads(zlib.decompress(f.read()))
        for i in data:
            data[i]['slim'] = new_labels[i]
        logger.info(' ' * 11 + "==> Finished!")
    else:
        new_labels = {}
        logger.info(' ' * 11 + f"==> Saving slimmed label cache to {slim_labels_path}")
        for i, item in data.items():
            new_labels[i] = filter_tiny_nf(np.clip(item['lab'], 0, 1).copy())
            data[i]['slim'] = new_labels[i]
        with slim_labels_path.open("wb") as f:
            f.write(zlib.compress(pickle.dumps(new_labels, pickle.HIGHEST_PROTOCOL)))
        logger.info(' ' * 11 + "==> Finished!")

    return data


def load_test_data_paths():
    data_dir = DATA_ROOT / "test_NF"
    path_list = list(data_dir.glob("*img.nii.gz"))
    dataset = {}
    for path in path_list:
        pid = int(path.name.split("-")[0])
        dataset[pid] = {"img_path": path, "lab_path": path.parent / path.name.replace("img", "mask")}
    return dataset

# Functions added during finetuning / training of DINs on Neurofibroma_UKE data
import SimpleITK as sitk

# Function to reorient an image to RAS
def reorient_to_rsa(image):
    # Reorient the image to RAS
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RSA")
    ras_image = orient_filter.Execute(image)
    return ras_image

# Function to resample an image to the desired spacing
def resample_image(image, spacing, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute new size based on the desired spacing
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For label masks
    else:
        resampler.SetInterpolator(sitk.sitkLinear)  # For intensity images
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    return resampler.Execute(image)

# Function to apply Z-score normalization
def z_score_normalization(image):
    array = sitk.GetArrayFromImage(image)
    mean = array.mean()
    std = array.std()
    normalized_array = (array - mean) / std
    return sitk.GetImageFromArray(normalized_array, isVector=image.GetNumberOfComponentsPerPixel() > 1)

def read_nii_uke_nf(file_name, spacing=(1.7, 1.7, 7.8), special=False, only_header=False, is_label=False):
    if is_label:
        out_dtype = np.uint8
    else:
        out_dtype = np.float32
        
    image = sitk.ReadImage(str(file_name))

    if only_header:
        return image.GetMetaDataKeys()

    # Reorient to RSA
    image = reorient_to_rsa(image)

    # Resample to desired spacing
    image = resample_image(image, spacing, is_label)
    
    if not is_label:
        # Apply Z-score normalization
        image = z_score_normalization(image)

    # Convert to numpy array
    data = sitk.GetArrayFromImage(image).astype(out_dtype)

    # Handle special case (e.g., flipping along an axis)
    if special:
        data = np.flip(data, axis=2)

    # Extract metadata similar to vh
    meta_data = {
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "size": image.GetSize(),
    }

    return meta_data, data

def load_data_uke_nf(logger, opt):
    # Define data directories
    images_dir = Path(opt.data_root) / "imagesTr"
    labels_dir = Path(opt.data_root) / "labelsTr"
    
    # List of image files
    path_list = sorted(images_dir.glob("*.nii.gz"))

    logger.info(' ' * 11 + f"==> Loading data ({len(path_list)} examples) ...")

    # Define cache file path
    cache_path = Path(opt.data_root) / "cache.pkl.gz"
    
    # Load data from cache if available
    if cache_path.exists():
        logger.info(' ' * 11 + f"==> Loading data cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            _data_cache = pickle.loads(data)
        logger.info(' ' * 11 + "==> Finished!")
        return _data_cache

    _data_cache = {}
    
     # Process each case
    for image_path in tqdm.tqdm(path_list):
        # Extract patient ID from the filename
        pid = image_path.stem.split(".")[0]
        
        # Load image volume
        header, volume = read_nii_uke_nf(file_name=image_path, spacing=opt.spacing, is_label=False)

        # Derive corresponding label path
        label_path = labels_dir / image_path.name
        # Load label volume
        _, label = read_nii_uke_nf(file_name=label_path, spacing=opt.spacing, is_label=True)
        
        # Ensure the volume and label shapes match
        assert volume.shape == label.shape, f"{volume.shape} vs {label.shape}"
        
        # Cache the data
        _data_cache[pid] = {
            "im_path": image_path.absolute(),
            "la_path": label_path.absolute(),
            "img": volume,
            "lab": label.astype(np.uint8),
            "pos": np.stack(np.where(label > 0), axis=1),
            "meta": header,
            "lab_rng": np.unique(label)
        }
        
    with cache_path.open("wb") as f:
        logger.info(' ' * 11 + f"==> Saving data cache to {cache_path}")
        cache_s = pickle.dumps(_data_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    logger.info(' ' * 11 + "==> Finished!")
    return _data_cache

def load_split_uke_nf(set_key, test_fold, fold_path, opt):
    # Define the folder containing the splits
    split_folder = Path(fold_path) / f"fold_{test_fold}"
    
    if set_key in ["train", "eval_online", "eval"]:
        # Define file paths for train and validation sets
        train_file = split_folder / "train_set.txt"
        val_file = split_folder / "val_set.txt"

        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Train or validation file missing in {split_folder}")

        # Read the file names from the text files
        with train_file.open("r") as f:
            train_set = [line.strip() for line in f.readlines()]

        with val_file.open("r") as f:
            val_set = [line.strip() for line in f.readlines()]

        # Form a DataFrame with columns: split, pid, remove
        data = []
        if set_key == "train":
            for pid in train_set:
                if pid not in val_set:
                    data.append({"split": 0, "pid": pid, "remove": 0})
            for pid in val_set:
                data.append({"split": test_fold, "pid": pid, "remove": 0})
        else:
            for pid in val_set:
                data.append({"split": test_fold, "pid": pid, "remove": 0})

        return pd.DataFrame(data)

    
    elif set_key == "test":
        # For the test set, return all files in the test directories
        test_images_dir = Path(opt.data_root) / f"imagesTs_{test_fold}"

        # Gather all test image filenames without extensions
        test_set = [path.stem.split(".")[0] for path in test_images_dir.glob("*.nii.gz")]
        
         # Form a DataFrame with columns: split, pid, remove
        data = []
        for pid in test_set:
            data.append({"split": 0, "pid": pid, "remove": 0})
        return pd.DataFrame(data)

    else:
        raise ValueError(f"`set_key` supports [train|eval_online|eval|test], got {set_key}")
    
def load_test_data_paths_uke_nf(opt):
    # Define test subset folders based on opt.test_subset
    test_images_dir = Path(opt.data_root) / f"imagesTs_{opt.test_subset}"
    test_labels_dir = Path(opt.data_root) / f"labelsTs_{opt.test_subset}"
    
    # List of image files in the test images directory
    path_list = sorted(test_images_dir.glob("*.nii.gz"))
    
    # Initialize dataset dictionary
    dataset = {}

    # Process each test case
    for image_path in path_list:
        # Extract patient ID from the filename
        pid = image_path.stem.split(".")[0]

        # Derive corresponding label path
        label_path = test_labels_dir / image_path.name

        # Add paths to the dataset
        dataset[pid] = {
            "img_path": image_path,
            "lab_path": label_path
        }

    return dataset
    