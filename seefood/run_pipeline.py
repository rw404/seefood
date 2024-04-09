import lightning as L
from augs import get_augs
from data import MyCustomDataset
from model import UfaNet
from torch.utils.data import DataLoader


def main():
    """Pipeline function."""
    train_tfms, test_tfms = get_augs()
    dataset_train = MyCustomDataset(
        mode="train",
        img_dir="candies/data/train",
        label_root_dir="candies",
        tfms=train_tfms,
    )
    dataset_val = MyCustomDataset(
        mode="val",
        img_dir="candies/data/train",
        label_root_dir="candies",
        tfms=test_tfms,
    )
    dataset_test = MyCustomDataset(
        mode="test",
        img_dir="candies/data/test",
        label_root_dir="candies",
        tfms=test_tfms,
    )

    dl_train = DataLoader(
        dataset_train,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    dl_val = DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True
    )
    dl_test = DataLoader(
        dataset_test,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    ufaNetPomenbshe = UfaNet()
    # Checkpoints
    MyModelCheckpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="./runs/",
        filename="best",
        monitor="val_average_acc",
        mode="max",
        save_top_k=1,
    )

    training = L.Trainer(
        max_epochs=10,
        accelerator="cpu",
        devices="auto",
        logger=False,
        callbacks=[MyModelCheckpoint],
    )

    training.fit(ufaNetPomenbshe, dl_train, dl_val)
    training.test(ckpt_path="./runs/best-v11.ckpt", dataloaders=dl_test)


if __name__ == "__main__":
    main()
