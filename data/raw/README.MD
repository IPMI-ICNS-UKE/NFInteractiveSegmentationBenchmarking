# README for `data/raw` Directory

This directory is intended for storing raw medical image datasets in **NIfTI format** (`.nii` or `.nii.gz`). The dataset should follow a structured naming convention to ensure compatibility with the processing pipeline.

## Directory Structure
```
data/raw/
│── imagesTr/                 # Training images (NIfTI format)
│── labelsTr_instance/        # Instance segmentation masks for training
│── imagesTs_1/               # Test images (NIfTI format)
│── labelsTs_1_instance/      # Instance segmentation masks for testing
```

## File Naming Conventions
- Each image in `imagesTr/` and `imagesTs/` should have a corresponding mask in `labelsTr_instance/` and `labelsTs_instance/`, respectively.
- The filenames of images and masks must match, e.g.:
  ```
  imagesTr/case_001.nii.gz      <-> labelsTr_instance/case_001.nii.gz
  imagesTs/case_002.nii.gz      <-> labelsTs_instance/case_002.nii.gz
  ```

## Data Format
- **Modality:** MRI, CT, or other volumetric medical imaging data.
- **Format:** `.nii` or `.nii.gz` (compressed NIfTI).

## Usage
Place your dataset inside this folder following the above structure.
