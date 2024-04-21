import numpy as np
import pytest
import torch


@pytest.fixture()
def test_img():
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def test_batch():
    return torch.randn(2, 3, 224, 224, dtype=torch.float32)
