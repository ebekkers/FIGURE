import os
import h5py
import torch
import requests
from torch.utils.data import Dataset
from torchvision import transforms

class FigureDataset(Dataset):
    """PyTorch Dataset for FIGURE benchmark. Downloads HDF5 files if missing."""

    BASE_URL = "https://github.com/ebekkers/FIGURE/raw/main/datasets/"  # GitHub URL for datasets

    def __init__(self, dataset_name, data_dir="figure_datasets", transform=None, download=True):
        """
        Args:
            dataset_name (str): Name of the dataset (e.g., 'FIGURE-Shape-B').
            transform (callable, optional): Transformations to apply to images.
            download (bool): Whether to download the dataset if not found.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, f"{dataset_name}.h5")
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
        """Downloads the dataset from GitHub if it's not already available."""
        os.makedirs(self.data_dir, exist_ok=True)
        url = f"{self.BASE_URL}{self.dataset_name}.h5"
        print(f"Downloading {self.dataset_name} from {url} to {self.data_dir}...")

        response = requests.get(url, stream=True, allow_redirects=True)  # Ensure GitHub redirects are handled
        if response.status_code == 200:
            with open(self.file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded {self.dataset_name} to {self.file_path}.")
        else:
            print(f"HTTP Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            raise RuntimeError(f"Failed to download {self.dataset_name}. URL might be incorrect or requires authentication.")

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
    dataset = FigureDataset("FIGURE-Shape-B", download=True)
    print(f"Dataset size: {len(dataset)} samples")

    # Example: Load a single sample
    image, points_r2, points_se2, label = dataset[0]
    print("Image shape:", image.shape)
    print("Label:", label)


    from torch.utils.data import DataLoader

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through batches
    for images, points_r2, points_se2, labels in dataloader:
        print("Batch images shape:", images.shape)
        print("Batch labels:", labels)
        break  # Show only first batch
