import os
from typing import Tuple

import albumentations
import pandas as pd
import torch
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MyCustomDataset(Dataset):
    """Класс датасета для подгрузки данных.

    Attributes:
        mode (str): Тип датасета: train, val, test
        img_dir (str):  Путь до директории с изображениями.
        tfms (albumentations.Compose): Трансформации изображений.
        cls_name_to_id (dict): Словарь отображение названия класса в его метку.
        {
            'twix' : 0,
            'snickers' : 1,
            'orbit': 2,
        }
        label_name (str): Название .csv файла с разметкой
        data_buf (pd.DataFrame): Данные из файла с разметкой в виде pd.DataFrame
    """

    def __init__(
        self,
        mode: str,
        img_dir: str,
        label_root_dir: str,
        tfms: albumentations.Compose,
        train_fraction: float = 0.8,
    ):
        """Инициализация объекта класса MyCustomDataset.

        Args:
            mode (str): Тип датасета: train, val, test.
            img_dir (str): Путь до директории с изображениями.
            label_root_dir (str): Путь до .csv файлов с разметкой.
            tfms (albumentations.Compose): Трансформации изображений.
            train_fraction (float, optional): Размер тренирочной выборки. Дефолтное значение: 0.8.
        """
        self.mode = mode
        self.img_dir = img_dir
        self.tfms = tfms
        self.cls_name_to_id = {
            "twix": 0,
            "snickers": 1,
            "orbit": 2,
        }

        label_name = None
        if self.mode == "train" or self.mode == "val":
            label_name = "train.csv"
        else:
            label_name = "test.csv"

        # получение полного пути до .csv файла с разметкой
        label_dir = os.path.join(label_root_dir, label_name)

        # чтение разметки из .csv файла в pd.Dataframe
        image_label_info = pd.read_csv(label_dir)

        self.data_buf = None
        if mode != "test":
            # разбиение данных на тренировочную и валидационную выборки
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

    def __len__(self) -> int:
        """Вычисление длины датасета.

        Returns:
            int: Длина датасета.
        """
        return self.data_buf.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получение изображения и метки класса по индексу.

        Args:
            index (int): Индекс элемента в датасете.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Изображение, метка класса
        """
        # чтение названия изображения и названия класса из датафрейма
        img_path, label = self.data_buf.iloc[index]

        # преобразовываем название класса в id класса
        label = self.cls_name_to_id[label]

        # получение полного пути до изображения
        img_path = os.path.join(self.img_dir, img_path)

        # чтение изображения по данному пути
        image = imread(img_path)

        # применение аугментаций к изображению
        image = self.tfms(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)
