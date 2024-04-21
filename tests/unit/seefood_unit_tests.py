import numpy as np
import torch
from torch.utils.data import DataLoader

from seefood.augs import get_augmentations_transformations
from seefood.data import MyCustomDataset
from seefood.model import UfaNet


def test_transforms_not_fail(test_img: np.ndarray):
    train_tsfm, test_tsfm = get_augmentations_transformations()
    train_tsfm(image=test_img)["image"]
    test_tsfm(image=test_img)["image"]


def test_transforms_return(test_img: np.ndarray):
    train_tsfm, test_tsfm = get_augmentations_transformations()
    train_transformed_img = train_tsfm(image=test_img)["image"]
    test_transformed_img = test_tsfm(image=test_img)["image"]

    assert train_transformed_img.dtype == torch.float32
    assert test_transformed_img.dtype == torch.float32
    assert train_transformed_img.shape == torch.Size([3, 224, 224])
    assert test_transformed_img.shape == torch.Size([3, 224, 224])


def test_init_model():
    UfaNet()


def test_model_forward(test_batch: torch.Tensor):
    model = UfaNet()
    model(test_batch)


def test_model_output_shape(test_batch: torch.Tensor):
    model = UfaNet()
    pred = model(test_batch)

    assert pred.dtype == torch.float32
    assert pred.shape == torch.Size([2, 3])


def test_init_datasets():
    train_transformations, test_transformations = get_augmentations_transformations()
    MyCustomDataset(
        mode="train",
        img_dir="tests/test_data/data/train",
        label_root_dir="tests/test_data",
        tfms=train_transformations,
    )

    MyCustomDataset(
        mode="val",
        img_dir="tests/test_data/data/train",
        label_root_dir="tests/test_data",
        tfms=test_transformations,
    )

    MyCustomDataset(
        mode="test",
        img_dir="tests/test_data/data/test",
        label_root_dir="tests/test_data",
        tfms=test_transformations,
    )


def test_init_dataloaders():
    train_transformations, test_transformations = get_augmentations_transformations()
    train_dataset = MyCustomDataset(
        mode="train",
        img_dir="tests/test_data/data/train",
        label_root_dir="tests/test_data",
        tfms=train_transformations,
    )

    val_dataset = MyCustomDataset(
        mode="val",
        img_dir="tests/test_data/data/train",
        label_root_dir="tests/test_data",
        tfms=test_transformations,
    )

    test_dataset = MyCustomDataset(
        mode="test",
        img_dir="tests/test_data/data/test",
        label_root_dir="tests/test_data",
        tfms=test_transformations,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    for _ in train_data_loader:
        pass

    val_data_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True
    )

    for _ in val_data_loader:
        pass

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    for _ in test_data_loader:
        pass
