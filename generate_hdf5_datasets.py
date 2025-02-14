from generate_single_dataset import generate_dataset

if __name__ == "__main__":
    dataset_names = ["FIGURE-Shape-B", "FIGURE-Shape-CB", "FIGURE-Shape-PI", "FIGURE-Shape-F"]

    for dataset_name in dataset_names:
        generate_dataset(dataset_name)
