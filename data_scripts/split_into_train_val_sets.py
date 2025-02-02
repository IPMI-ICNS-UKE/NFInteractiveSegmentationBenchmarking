import os
from sklearn.model_selection import KFold


def create_k_fold_splits(input_path, split_path, num_folds=5):
    """
    Create k-fold cross-validation splits for files in the input path.

    Args:
        input_path (str): Path to the directory containing the files.
        split_path (str): Path to the directory where split files will be saved.
        num_folds (int): Number of folds for cross-validation.
    """
    # Ensure the split directory exists
    os.makedirs(split_path, exist_ok=True)

    # Get file names without extensions
    file_names = [os.path.splitext(os.path.splitext(f)[0])[0] for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    # Sort file names for consistency
    file_names = sorted(file_names)

    # Create k-fold cross-validation splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(file_names), start=1):
        fold_dir = os.path.join(split_path, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split the data
        train_set = [file_names[i] for i in train_idx]
        val_set = [file_names[i] for i in val_idx]

        # Write train and validation sets to files
        with open(os.path.join(fold_dir, "train_set.txt"), "w") as train_file:
            train_file.write("\n".join(train_set))

        with open(os.path.join(fold_dir, "val_set.txt"), "w") as val_file:
            val_file.write("\n".join(val_set))

        print(f"Fold {fold_idx}: Train set and validation set written to {fold_dir}")

    print("All folds processed.")


if __name__ == "__main__":
    # Set paths and choose the number of folds
    input_path = "../data/raw/imagesTr"
    split_path = "../data/splits"
    num_folds = 5

    create_k_fold_splits(input_path, split_path, num_folds)
