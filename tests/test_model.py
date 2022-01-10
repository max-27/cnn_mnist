from src.models.model import MyAwesomeModel
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
import pytest


ROOT_PATH = Path(__name__).resolve().parents[1]
DATA_PATH = os.path.join(ROOT_PATH, "data/processed")


@pytest.mark.parametrize("data_sample, cnn_output_size", [(torch.randint(0, 255, size=(1, 28, 28)), 196),
                                                          (torch.randint(0, 255, size=(1, 40, 40)), 400),
                                                          (torch.randint(0, 255, size=(1, 50, 50)), 576),
                                         ])
def test_model(data_sample, cnn_output_size) -> None:
    model = MyAwesomeModel(image_width=data_sample.size()[-1])
    output = model.forward(data_sample.unsqueeze(1).float())
    output_cnn = model.forward_visualize(data_sample.unsqueeze(1).float())
    assert output.shape == torch.Size([1, 10]), f"Expected output size {torch.Size([1, 10])} but got {output.shape}"
    assert output_cnn.shape == torch.Size([cnn_output_size]), f"Expected output of flatten cnn layer " \
                                                              f"{cnn_output_size} but got {output_cnn.shape} "


def test_error_on_wrong_shape() -> None:
    x = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match=f"Gray scale image with one channel expected: Got {x.shape[1]} channels!"):
        model = MyAwesomeModel()
        model.forward(x)
