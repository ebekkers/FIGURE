import os
import h5py
import torch
import requests
from torch.utils.data import Dataset
from torchvision import transforms

class FigureDataset(Dataset):
    """PyTorch Dataset for FIGURE benchmark. Downloads HDF5 files if missing."""

    BASE_URL = "https://staff.fnwi.uva.nl/e.j.bekkers/FIGURE/"

    def __init__(self, dataset_name, split="train", data_dir="figure_datasets", transform=None, download=True):
        """
        Args:
            dataset_name (str): Name of the dataset (e.g., 'FIGURE-Shape-B').
            split (str): Dataset split - "train" (default), "test", or "test-bias".
            data_dir (str): Local directory to store datasets.
            transform (callable, optional): Transformations to apply to images.
            download (bool): Whether to download the dataset if not found.
        """
        assert split in ["train", "test", "test-bias"], "Invalid split! Choose from 'train', 'test', or 'test-bias'."
        
        self.dataset_name = dataset_name
        self.split = split
        self.data_dir = data_dir

        # Determine filename based on split
        file_suffix = "" if split == "train" else f"-{split}"
        self.file_name = f"{dataset_name}{file_suffix}.h5"
        self.file_path = os.path.join(self.data_dir, self.file_name)

        self.transform = transform

        if download and not os.path.exists(self.file_path):
            self._download_dataset()

        # Open the HDF5 file
        self.h5file = h5py.File(self.file_path, "r")
        self.images = self.h5file["images"]
        self.points_r2 = self.h5file["points_r2"]
        self.points_se2 = self.h5file["points_se2"]
        self.labels = self.h5file["labels"]

    def _download_dataset(self):
        """Downloads the dataset from the official server if it's not already available."""
        os.makedirs(self.data_dir, exist_ok=True)
        url = f"{self.BASE_URL}{self.file_name}"
        print(f"Downloading {self.dataset_name} ({self.split}) from {url} to {self.data_dir}...")

        response = requests.get(url, stream=True, allow_redirects=True)
        if response.status_code == 200:
            with open(self.file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded {self.dataset_name} ({self.split}) to {self.file_path}.")
        else:
            print(f"HTTP Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            raise RuntimeError(f"Failed to download {self.dataset_name} ({self.split}). URL might be incorrect.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset."""
        image = self.images[idx]  # Image (H, W, 3)
        points_r2 = self.points_r2[idx]  # 2D keypoints
        points_se2 = self.points_se2[idx]  # SE(2) keypoints
        label = self.labels[idx]  # Class label (0 = up, 1 = down)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0,1]
        points_r2 = torch.tensor(points_r2, dtype=torch.float32)
        points_se2 = torch.tensor(points_se2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, points_r2, points_se2, label

    def close(self):
        """Closes the HDF5 file."""
        self.h5file.close()

    def __del__(self):
        """Ensure the file is closed when the dataset object is deleted."""
        self.close()



if __name__ == "__main__":
    dataset = FigureDataset("FIGURE-Shape-B", split="train", download=True)
    print(f"Training set size: {len(dataset)} samples")

    test_dataset = FigureDataset("FIGURE-Shape-B", split="test", download=True)
    print(f"Test set size: {len(test_dataset)} samples")

    test_bias_dataset = FigureDataset("FIGURE-Shape-CB", split="test-bias", download=True)
    print(f"Bias-swapped test set size: {len(test_bias_dataset)} samples")

    from torch.utils.data import DataLoader

    # Create DataLoader for training set
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, points_r2, points_se2, labels in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch points_r2 shape:", points_r2.shape)
        print("Batch points_se2 shape:", points_se2.shape)
        print("Batch labels:", labels)
        break



