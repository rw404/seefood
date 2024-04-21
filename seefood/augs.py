from typing import Tuple

import albumentations as A
import albumentations.pytorch as Ap


def get_augmentations_transformations() -> Tuple[A.Compose, A.Compose]:
    """Создать преобразования и аугментации для наборов данных

    Returns:
        Tuple[A.Compose, A.Compose]: Преобразования(аугментации) для тренировочной выборки и преобразования для тестовой выборки
    """

    # Создаем преобразования для тренировочной выборки(преобразования + аугментации)
    train_transformations = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Flip(),
            A.RandomBrightnessContrast(),
            A.Rotate(limit=10),
            A.Normalize(),
            Ap.ToTensorV2(),
        ]
    )

    # Создаем преобразования для тестировочной выборки(только преобразования)
    test_transformations = A.Compose(
        [A.Resize(height=224, width=224), A.Normalize(), Ap.ToTensorV2()]
    )
    return train_transformations, test_transformations
