import os
from unittest import TestCase

import pytest
from omegaconf import OmegaConf

from src.models.train_model import train
from tests import _PATH_DATA

data_path = os.path.join(_PATH_DATA, "processed/test")
config = OmegaConf.create(
    {
        "experiment": {
            "exp1": {
                "model": {
                    "image_width": 28,
                    "image_height": 28,
                    "con_layer1": {
                        "input_channel": 1,
                        "output_channel": 4,
                        "kernel_size": 3,
                        "padding": 1,
                        "stride": 1,
                    },
                    "max_pool1": {"kernel_size": 2, "stride": 2},
                    "con_layer2": {
                        "input_channel": 1,
                        "output_channel": 4,
                        "kernel_size": 3,
                        "padding": 1,
                        "stride": 1,
                    },
                    "max_pool2": {"kernel_stride": 2, "stride": 2},
                    "output_dim": 10,
                },
                "training": {
                    "data_path": f"{data_path}",
                    "lr": 0.001,
                    "batch_size": 64,
                    "epochs": 1,
                },
            }
        }
    }
)

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"


class TestTraining(TestCase):
    @pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
    def test_training(self):
        with self.assertLogs() as captured:
            train(config)
        self.assertEqual(captured.records[-1].getMessage(), "Finished training and saved model!")
