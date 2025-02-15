import h5py
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from figure_shape import Shape  # Ensure this is correctly implemented

def generate_gif(images, labels, dataset_name, save_dir, num_samples=10):
    """Generates an animated GIF with sample images from each class."""
    os.makedirs(f"{save_dir}/visualizations", exist_ok=True)

    classes = ["up", "down"]
    frames = {cls: [] for cls in classes}

    for cls_index, cls_label in enumerate(classes):
        count = 0
        for i in range(len(labels)):
            if labels[i] == cls_index and count < num_samples:
                img = Image.fromarray(images[i])
                frames[cls_label].append(img)
                count += 1
            if count >= num_samples:
                break

    # Save GIFs for each class
    gif_path = f"{save_dir}/visualizations/{dataset_name}.gif"
    frames["up"][0].save(
        gif_path,
        save_all=True,
        append_images=frames["up"] + frames["down"],
        duration=500,  # 500ms per frame
        loop=0
    )
    print(f"Saved animated GIF: {gif_path}")

def generate_dataset(dataset_name, num_samples_per_class=5000, image_size=64, is_test=False, bias_swapped=False):
    """Generates and saves a shape classification dataset to an HDF5 file."""
    
    save_dir = "datasets"
    os.makedirs(save_dir, exist_ok=True)

    # Adjust dataset name for test sets
    if is_test:
        dataset_name += "-test-bias" if bias_swapped else "-test"
    
    save_path = f"{save_dir}/{dataset_name}.h5"

    # Define classes
    classes = ["up", "down"]
    num_classes = len(classes)

    # Define dataset shapes
    num_samples = num_samples_per_class * num_classes
    images_shape = (num_samples, image_size, image_size, 3)
    points_r2_shape = (num_samples, 13, 2)
    points_se2_shape = (num_samples, 9, 3)
    labels_shape = (num_samples,)

    # Dataset-specific configurations
    config = {
        "FIGURE-Shape-B": {"color_bias": False, "pose_variation": False},
        "FIGURE-Shape-CB": {"color_bias": True, "pose_variation": False},
        "FIGURE-Shape-PI": {"color_bias": False, "pose_variation": True},
        "FIGURE-Shape-F": {"color_bias": True, "pose_variation": True},
    }[dataset_name.replace("-test", "").replace("-bias", "")]  # Remove test suffix to get original name

    print(f"Generating dataset: {dataset_name}")

    # Create HDF5 file
    with h5py.File(save_path, "w") as h5file:
        # Create datasets
        images_dset = h5file.create_dataset("images", shape=images_shape, dtype="uint8", compression="gzip")
        points_r2_dset = h5file.create_dataset("points_r2", shape=points_r2_shape, dtype="float32", compression="gzip")
        points_se2_dset = h5file.create_dataset("points_se2", shape=points_se2_shape, dtype="float32", compression="gzip")
        labels_dset = h5file.create_dataset("labels", shape=labels_shape, dtype="int8")

        metadata = {
            "num_samples_per_class": num_samples_per_class,
            "image_shape": images_shape,
            "points_r2_shape": points_r2_shape,
            "points_se2_shape": points_se2_shape,
            "label_mapping": {0: "up", 1: "down"},
        }

        index = 0
        gif_samples = {cls: [] for cls in classes}

        for class_index, class_label in enumerate(classes):
            count = 0  # Track how many images we store per class for the GIF
            for _ in tqdm(range(num_samples_per_class), desc=f"Generating {dataset_name} ({class_label})"):

                # Set color probabilities
                if config['color_bias']:
                    if bias_swapped:
                        color_probabilities = {"up": [0.0, 0.0, 0.5, 0.5], "down": [0.5, 0.5, 0.0, 0.0]}  # Swapped!
                    else:
                        color_probabilities = {"up": [0.5, 0.5, 0.0, 0.0], "down": [0.0, 0.0, 0.5, 0.5]}  # Default bias
                else:
                    color_probabilities = {"up": [1, 0, 0, 0], "down": [1, 0, 0, 0]}  # No bias

                shift_min_max = 2
                figure = Shape(
                    torso_colors=["red", "green", "blue", "gold"],
                    color_probabilities=color_probabilities,
                )

                g = [
                    np.random.uniform(-shift_min_max, shift_min_max) if config["pose_variation"] else 0.0,
                    np.random.uniform(-shift_min_max, shift_min_max) if config["pose_variation"] else 0.0,
                    np.random.uniform(0, 2 * np.pi) if config["pose_variation"] else 0.0
                ]

                figure.resample(g=g, pose_class=class_label)

                # Render image and get keypoints
                image, points_r2, points_se2 = figure.render(image_size=image_size)

                # Store data in HDF5
                images_dset[index] = image
                points_r2_dset[index] = points_r2
                points_se2_dset[index] = points_se2
                labels_dset[index] = class_index  # 0: up, 1: down

                # Collect first 10 samples per class for the GIF
                if count < 10:
                    gif_samples[class_label].append(image)
                    count += 1

                index += 1

        # Store metadata
        h5file.attrs["metadata"] = json.dumps(metadata)

    # Generate and save sample GIF
    generate_gif(
        images=gif_samples["up"] + gif_samples["down"],  # Merge up + down images
        labels=[0] * 10 + [1] * 10,  # Corresponding labels
        dataset_name=dataset_name,
        save_dir=save_dir
    )

    print(f"Dataset {dataset_name} saved successfully to {save_path}!")

if __name__ == "__main__":
    dataset_names = ["FIGURE-Shape-B", "FIGURE-Shape-CB", "FIGURE-Shape-PI", "FIGURE-Shape-F"]

    for dataset_name in dataset_names:
        # Generate training dataset
        generate_dataset(dataset_name)

        # Generate test dataset (20% of training size)
        generate_dataset(dataset_name, is_test=True)

        # Generate bias-swapped test set for CB & F
        if "CB" in dataset_name or "F" in dataset_name:
            generate_dataset(dataset_name, is_test=True, bias_swapped=True)
