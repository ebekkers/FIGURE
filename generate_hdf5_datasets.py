from generate_single_dataset import generate_dataset

if __name__ == "__main__":
    dataset_names = ["FIGURE-Shape-B", "FIGURE-Shape-CB", "FIGURE-Shape-PI", "FIGURE-Shape-F"]

    for dataset_name in dataset_names:
        # Generate training dataset
        generate_dataset(dataset_name, num_samples_per_class=1000)

        # Generate test dataset (20% of training size)
        generate_dataset(dataset_name, is_test=True, num_samples_per_class=1000)

        # Generate bias-swapped test set for CB & F
        if "-CB" in dataset_name or "-F" in dataset_name:
            generate_dataset(dataset_name, is_test=True, bias_swapped=True, num_samples_per_class=1000)