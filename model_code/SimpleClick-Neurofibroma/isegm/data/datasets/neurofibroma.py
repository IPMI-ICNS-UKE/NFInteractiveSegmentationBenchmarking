from pathlib import Path
from PIL import Image as PILImage
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import numpy as np


class NeurofibromaDataset(ISDataset):
    def __init__(self, dataset_path, split, 
                 images_dir_name="imagesTr", 
                 masks_dir_name="labelsTr_instance", 
                 **kwargs):
        """
        Initialize the NeurofibromaDataset instance.

        Args:
            dataset_path (str): Path to the root directory of the dataset.
            split (str): Path to the split file specifying the dataset subset.
            images_dir_name (str): Subdirectory name for images. Defaults to 'imagesTr'.
            masks_dir_name (str): Subdirectory name for masks. Defaults to 'labelsTr_instance'.
            **kwargs: Additional arguments passed to the ISDataset initializer.
        """
        super(NeurofibromaDataset, self).__init__(**kwargs)
        
        # Get paths
        dataset_path = Path(dataset_path)
        self.split = Path(split)  # Path to the split file
        self._images_path = dataset_path / images_dir_name
        self._masks_path = dataset_path / masks_dir_name
        
        # Validate paths
        if not self._images_path.exists() or not self._images_path.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self._images_path}")
        if not self._masks_path.exists() or not self._masks_path.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {self._masks_path}")
        if not self.split.exists():
            raise FileNotFoundError(f"Split file not found: {self.split}")

        # Read file names from the split file
        with open(self.split, 'r') as file:
            file_names = [line.strip() for line in file.readlines()]

        # Collect all matching .jpg and .png files for images and masks
        self.dataset_samples = []
        self._masks_paths = {}
        for name in file_names:
            image_files = sorted((self._images_path / name).glob("*.jpg"))
            mask_files = sorted((self._masks_path / name).glob("*.png"))

            # Collect matching .png mask files
            self.dataset_samples.extend(image_files)
            for mask_file in mask_files:
                self._masks_paths[mask_file.stem] = mask_file

    def get_sample(self, index) -> DSample:
        """
        Retrieve a dataset sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            DSample: A dataset sample containing image, mask, and metadata.
        """
        image_path = self.dataset_samples[index]
        mask_path = self._masks_paths[image_path.stem]

        # Read image and mask
        image = PILImage.open(image_path).convert("RGB")
        mask = PILImage.open(mask_path).convert("P")

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Extract unique object IDs, excluding background (ID=0)
        objects_ids = np.unique(mask)
        objects_ids = [x for x in objects_ids if x != 0] 

        return DSample(image, mask, objects_ids=objects_ids, ignore_ids=[-1], sample_id=index)
