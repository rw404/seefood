import lightning as L
from augs import get_augmentations_transformations
from data import MyCustomDataset
from model import UfaNet
from torch.utils.data import DataLoader


def main() -> None:
    """Создание модели, обучение и тестирование

    1. Создаются аугментации(трансформации) для данных;
    2. Создаются наборы данных(обучение/валидация/тестирование);
    3. Оборачиваются наборы данных в итерируемые объекты(Даталоадеры);
    4. Создается модель;
    5. Создается учитель модели с дополнительными функциями:
        * Создание контрольных сохранений после эпохи;
    6. Обучается модель(используются обучающая + валидационная выборки);
    7. Лучшая модель проверяется на тестовой выборке.
    """

    # Определяем аугментации данных для тренировки и теста
    train_transformations, test_transformations = get_augmentations_transformations()

    # Создаем датасет для обучения
    # Важно указать:
    # - mode = "train"
    # - tfms = train_transformations
    train_dataset = MyCustomDataset(
        mode="train",
        img_dir="data/train",
        label_root_dir="data",
        tfms=train_transformations,
    )

    # Создаем датасет для валидации(синтетическое тестирование)
    # Важно указать:
    # - mode = "val"
    # - tfms = test_transformations
    val_dataset = MyCustomDataset(
        mode="val",
        img_dir="data/train",
        label_root_dir="data",
        tfms=test_transformations,
    )

    # Создаем датасет для тестирования
    # Важно указать:
    # - mode = "test"
    # - tfms = test_transformations
    test_dataset = MyCustomDataset(
        mode="test",
        img_dir="data/test",
        label_root_dir="data",
        tfms=test_transformations,
    )

    # Оборачиваем датасеты в итерируемые объекты torch.utils.data.DataLoader
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Создаем модель
    neural_candies_classifier = UfaNet()

    # Создадим возможность сохранения лучшей модели после каждой эпохи обучения
    # В ./runs/ будет лежать best.ckpt модель по итогам валидационной оценки точности
    BestAccuracyCheckpointProvider = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="./runs/",
        filename="best",
        monitor="val_average_acc",
        mode="max",
        save_top_k=1,
    )

    # Создаем учителя модели:
    # - 10 эпох
    # - обучение будет на 'cpu'
    # - в качестве дополнительных функций будут только контрольные сохранения
    training = L.Trainer(
        max_epochs=10,
        accelerator="cpu",
        devices="auto",
        logger=False,
        callbacks=[BestAccuracyCheckpointProvider],
    )

    # Запускаем обучение на train_dataset с проверкой качества на val_dataset
    training.fit(neural_candies_classifier, train_data_loader, val_data_loader)

    # Запускаем тестирование лучшей модели на test_dataset
    training.test(ckpt_path="./runs/best.ckpt", dataloaders=test_data_loader)


if __name__ == "__main__":
    main()
