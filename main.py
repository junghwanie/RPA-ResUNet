import os
import numpy as np
import argparse
from dataloader import LoadDataSet, get_train_transform, LoadTestSet, get_test_transform
from torch.utils.data import random_split, DataLoader
from solver import Solver


def main(config):

    base_dir = "data/"
    train_dir = os.path.join(base_dir, "stage1_train")
    test_dir = os.path.join(base_dir, "stage1_test")

    train_dataset = LoadDataSet(train_dir, transform=get_train_transform())
    test_dataset = LoadTestSet(test_dir, transform=get_test_transform())

    # Split the dataset
    train_size = int(np.round(train_dataset.__len__() * (1 - config.split_ratio), 0))
    valid_size = int(np.round(train_dataset.__len__() * config.split_ratio, 0))

    train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

    # Set data loader
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, config.batch_size)
    test_loader = DataLoader(test_dataset, config.batch_size)

    # Solver for training and testing ResUNet
    # Add to mode parameter for testing
    solver = Solver(train_loader, valid_loader, test_loader, config)

    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPA-ResUNet")
    parser.add_argument(
        "--split_ratio", type=float, default=0.25, help="Validation split ratio"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--workers", type=int, default=1, help="num_workers")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--alpha", type=int, default=5)
    config = parser.parse_args()

    main(config)