# README for `data/processed` Directory

This directory contains preprocessed medical image datasets, where each 3D volume has been converted into **2D slices** and stored in separate folders.

## Directory Structure
```
data/processed/
│── imagesTr/                 # Training image slices (JPG format)
│   ├── case_001/             # Slices for case 001
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   ├── ...
│── labelsTr_instance/        # Instance segmentation mask slices for training (PNG format)
│   ├── case_001/
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   ├── ...
│── imagesTs/                 # Test image slices (JPG format)
│   ├── case_002/
│   │   ├── 00000.jpg
│   │   ├── ...
│── labelsTs_instance/        # Instance segmentation mask slices for testing (PNG format)
│   ├── case_002/
│   │   ├── 00000.png
│   │   ├── ...
```

## File Naming Conventions
- Each case (patient scan) is stored as a **separate folder** containing **2D slices**.
- Image slices are saved as **JPG** (`.jpg`).
- Label mask slices are saved as **PNG** (`.png`).
- Each slice follows a **zero-padded numeric format**, e.g.:
  ```
  imagesTr/case_001/00000.jpg      <-> labelsTr_instance/case_001/00000.png
  imagesTs/case_002/00008.jpg      <-> labelsTs_instance/case_002/00008.png
  ```

## Data Format
- **Images:** Stored as **JPG** for efficient compression.
- **Labels:** Stored as **PNG** to preserve segmentation mask integrity.
- **Slice Order:** Slices are stored in the order they appear in the original NIfTI file.

## Usage
This folder is automatically generated from the `data/raw/` directory using the preprocessing pipeline. Place your processed dataset here following the above structure.
