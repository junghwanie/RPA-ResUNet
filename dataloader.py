import os
import numpy as np
import albumentations as A
import torch
import random
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from skimage import io, transform


def get_train_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            ToTensorV2(),
        ]
    )


class LoadDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_train_transform()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], "images/")
        mask_folder = os.path.join(self.path, self.folders[idx], "masks/")
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype("float32")
        img = transform.resize(img, (256, 256))
        mask = self.get_mask(mask_folder, 256, 256).astype("float32")

        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        mask = torch.permute(mask, (2, 0, 1))
        return (img, mask)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        return mask
    

def get_test_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class LoadTestSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_test_transform()

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], "images/")
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype("float32")
        img = transform.resize(img, (256, 256))

        augmented = self.transforms(image=img)
        img = augmented["image"]
        return img