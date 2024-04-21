import lightning as L
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.nn import functional as F


class UfaNet(L.LightningModule):
    """Модель классификации конфет."""

    def __init__(self, input_shape=(3, 224, 224), num_classes=50):
        """Инициализация полей класса, необходимых для обучения.

        Args:
            input_shape (tuple, optional): Размер входного изображения
            num_classes (int, optional): Количество предсказываемых классов
        """

        super().__init__()

        # Архитектура сверточной модели (3х224х224) -> (3)
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1),  # 224 -> 224-5+1 + 2*1 = 220
            nn.MaxPool2d(kernel_size=2),  # 220 -> 110
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1),  # 112 -> 112-5+1 = 108
            nn.MaxPool2d(kernel_size=2),  # 108 -> 54
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1),  # 54 -> 54-5+1= 50
            nn.MaxPool2d(kernel_size=2),  # 50 -> 25
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=1),  # 25 -> 25-5+1= 21
            nn.MaxPool2d(kernel_size=2),  # 21 -> 11
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),  # 11 -> 11-5+1= 7
            nn.MaxPool2d(kernel_size=2),  # 7 -> 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        # Confusion матрица
        self.cm = np.zeros((3, 3))
        self.confusion_matrix = lambda y_logit, y: confusion_matrix(
            y.flatten().detach().cpu().numpy(),
            y_logit.argmax(dim=-1).flatten().detach().cpu().numpy(),
            labels=[0, 1, 2],
        )

        # Инициализация функции потерь
        self.loss = F.cross_entropy

        # Инициализация метрики
        self.accuracy = lambda y_logit, y: accuracy_score(
            y.flatten().detach().cpu().numpy(),
            y_logit.argmax(dim=-1).flatten().detach().cpu().numpy(),
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        """Инициализация оптимизатора."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)

        return [optimizer]

    def forward(self, x):
        """Применение модели.

        Args:
            x (torch.tensor): Входной тензор

        Returns:
            torch.tensor: Результат работы модели
        """

        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        """Шаг тренировки.

        Args:
            batch (tuple): Входные данные и лейблы
            batch_idx (int): Номер батча

        Returns:
            dict: Значения метрики и ошибки
        """

        # Считываем данные и лейблы
        x, y = batch

        # Применяем модель к данным
        y_logit = self(x)

        # Считаем ошибку
        loss = self.loss(y_logit, y)

        # Считаем мтерику
        acc = self.accuracy(y_logit=y_logit, y=y)

        # Сохраняем значения ошибки и метрики
        output = {"loss": loss, "acc": acc}
        self.training_step_outputs.append(output)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=True)

        return output

    def validation_step(self, batch, batch_idx):
        """Шаг валидации.

        Args:
            batch (tuple): Входные данные и лейблы
            batch_idx (int): Номер батча

        Returns:
            dict: Значения метрики и ошибки
        """

        # Считываем данные и лейблы
        x, y = batch

        # Применяем модель к данным
        y_logit = self(x)

        # Считаем ошибку
        loss = self.loss(y_logit, y)

        # Считаем метрику
        val_acc = self.accuracy(y_logit=y_logit, y=y)

        # Изменяем confusion матрицу
        self.cm += self.confusion_matrix(y=y, y_logit=y_logit)

        # Сохраняем значения ошибки и метрики
        output = {"val_loss": loss, "val_acc": val_acc}
        self.validation_step_outputs.append(output)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=False)

        return output

    def test_step(self, batch, batch_idx):
        """Шаг Тестирования.

        Args:
            batch (tuple): Входные данные и лейблы
            batch_idx (int): Номер батча

        Returns:
            dict: Значения метрики и ошибки
        """

        # Аналогичен шагу валидации, но с тестовыми данными
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    def on_train_epoch_end(self) -> None:
        """Конец эпохи обучения."""

        # Сохранение метрик эпохи обучения
        avg_loss = torch.stack(
            [x["loss"].clone().detach() for x in self.training_step_outputs]
        )
        avg_loss = avg_loss.mean()

        avg_acc = torch.stack(
            [torch.tensor(x["acc"]) for x in self.training_step_outputs]
        )
        avg_acc = avg_acc.mean()

        Accuracy = 100 * avg_acc.item()

        self.training_step_outputs.clear()

        print(f"| Train_loss: {avg_loss:.5f} Train_acc: {Accuracy:.2f}%")

        self.log(
            "train_average_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "train_average_acc", avg_acc, prog_bar=True, on_epoch=True, on_step=False
        )

    def on_validation_epoch_end(self) -> None:
        """Конец эпохи валидации."""
        # Сохранение метрик валидации эпохи валидации
        avg_loss = torch.stack(
            [x["val_loss"].clone().detach() for x in self.validation_step_outputs]
        ).mean()
        avg_acc = torch.stack(
            [torch.tensor(x["val_acc"]) for x in self.validation_step_outputs]
        ).mean()
        Accuracy = 100 * avg_acc.item()

        self.validation_step_outputs.clear()

        print(
            f"[Epoch {self.trainer.current_epoch:3}] "
            + f"Val_loss: {avg_loss:.5f} "
            + f"Val_accuracy: {Accuracy:.2f}%",
            end=" ",
        )

        self.log(
            "val_average_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "val_average_acc", avg_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        print()
        print(self.cm)
        self.cm = np.zeros((3, 3))

    def on_test_epoch_end(self) -> None:
        """Конец эпохи тестирования."""
        return self.on_validation_epoch_end()
