from generate_single_dataset import generate_dataset
import os

if __name__ == "__main__":
    dataset_names = ["FIGURE-Shape-B", "FIGURE-Shape-CB", "FIGURE-Shape-PI", "FIGURE-Shape-F"]
    for color_consistency in [0.25, 0.5, 0.75, 0.9, 0.99, 1.0]:
        consistency_str = f"{color_consistency:.10f}".rstrip("0")
        if consistency_str.endswith("."):
            consistency_str += "0"
        save_dir = os.path.join("datasets", consistency_str)

        for dataset_name in dataset_names:
            # Generate training dataset
            generate_dataset(dataset_name, num_samples_per_class=1000, color_consistency=color_consistency, save_dir=save_dir)

            # Generate test dataset 
            generate_dataset(dataset_name, is_test=True, num_samples_per_class=1000, color_consistency=color_consistency, save_dir=save_dir)

            # Generate bias-swapped test set for CB & F
            if "-CB" in dataset_name or "-F" in dataset_name:
                generate_dataset(dataset_name, is_test=True, bias_swapped=True, num_samples_per_class=1000, color_consistency=color_consistency, save_dir=save_dir)