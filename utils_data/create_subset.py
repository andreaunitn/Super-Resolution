import random
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def create_dataset_subset(source_dir, dest_dir, percentage, seed=42):

    """
    Randomly selects a percentage of PNG files from a source directory and its
    subdirectories, then copies them to a new destination directory.

    Args:
        source_dir (str): The top-level directory containing the original full dataset.
        dest_dir (str): The directory where the subset will be saved.
        percentage (float): The percentage of files to select (e.g., 0.15 for 15%).
        seed (int): A random seed for reproducibility.
    """

    random.seed(seed)

    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory created at: {dest_path}")

    all_png_files = list(source_path.rglob('*.png'))

    if not all_png_files:
        print(f"Error: No .png files found in {source_path} or its subdirectories. Please check the path.")
        return

    print(f"Found {len(all_png_files)} total .png files.")

    num_to_select = int(len(all_png_files) * percentage)
    if num_to_select == 0:
        print(f"Error: Percentage {percentage:.2f}% is too low to select any files from a pool of {len(all_png_files)}. Please increase the percentage.")
        return
        
    print(f"Selecting {num_to_select} files ({percentage:.1%})...")

    selected_files = random.sample(all_png_files, num_to_select)
    print("File selection complete.")

    print(f"Copying {len(selected_files)} files to {dest_path}...")
    for file_path in tqdm(selected_files, desc="Copying files"):
        shutil.copy(file_path, dest_path / file_path.name)

    print("\nSubset creation complete!")
    print(f"Total files in destination: {len(list(dest_path.glob('*.png')))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creation of the training dataset from LSDIR")
    parser.add_argument("--source_dataset_dir", type=str, default=None, help="Path to the LSDIR dataset")
    parser.add_argument("--destination_dataset_dir", type=str, default=None, help="Where to save the subset")
    parser.add_argument("--subset_percentage", type=float, default=0.15, help="Percentage of the dataset to take")
    args = parser.parse_args()

    create_dataset_subset(args.source_dataset_dir, args.destination_dataset_dir, args.subset_percentage)