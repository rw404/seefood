import albumentations as A
import albumentations.pytorch as Ap


def get_augs():
    """Get train augs."""
    train_tfms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Flip(),
            A.RandomBrightnessContrast(),
            A.Rotate(limit=10),
            A.Normalize(),
            Ap.ToTensorV2(),
        ]
    )

    test_tfms = A.Compose(
        [A.Resize(height=224, width=224), A.Normalize(), Ap.ToTensorV2()]
    )
    return train_tfms, test_tfms
