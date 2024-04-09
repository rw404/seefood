import os

import pandas as pd
import torch
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MyCustomDataset(Dataset):
    """Dataset class."""

    def __init__(
        self,
        mode: str,
        img_dir: str,
        label_root_dir: str,
        tfms,
        train_fraction: float = 0.8,
    ):
        """Init class."""
        self.mode = mode
        self.img_dir = img_dir
        self.tfms = tfms

        label_name = None
        if self.mode == "train" or self.mode == "val":
            label_name = "train.csv"
        else:
            label_name = "test.csv"
        label_dir = os.path.join(label_root_dir, label_name)

        image_label_info = pd.read_csv(label_dir)

        self.data_buf = None
        if mode != "test":
            train_subset, val_subset = train_test_split(
                image_label_info,
                train_size=train_fraction,
                random_state=42,
                stratify=image_label_info.label.array,
            )

            if mode == "train":
                self.data_buf = train_subset
                del val_subset
            else:
                self.data_buf = val_subset
                del train_subset

        else:
            self.data_buf = image_label_info

    def __len__(self):
        """Get dataset length."""
        return self.data_buf.shape[0]

    def __getitem__(self, index):
        """Get dataset element."""
        img_path, label = self.data_buf.iloc[index]

        # img_path = self.img_dir + "/" + img_path
        img_path = os.path.join(self.img_dir, img_path)

        # read image
        image = imread(img_path)

        # image = Image.fromarray(image)

        image = self.tfms(image=image)["image"]

        if label == "twix":
            label = 0
        elif label == "snickers":
            label = 1
        elif label == "orbit":
            label = 2

        return image, torch.tensor(label, dtype=torch.long)
