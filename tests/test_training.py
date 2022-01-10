from unittest import TestCase
from src.models.train_model import train
from omegaconf import OmegaConf
import os


config = OmegaConf.create({'experiment':
                          {'exp1':
                               {'model':
                                    {'image_width': 28, 'image_height': 28,
                                     'con_layer1':
                                         {'input_channel': 1, 'output_channel': 4, 'kernel_size': 3, 'padding': 1,
                                          'stride': 1},
                                     'max_pool1': {'kernel_size': 2, 'stride': 2},
                                     'con_layer2': {'input_channel': 1, 'output_channel': 4, 'kernel_size': 3,
                                                    'padding': 1, 'stride': 1},
                                     'max_pool2': {'kernel_stride': 2, 'stride': 2}, 'output_dim': 10},
                                'training':
                                    {'data_path': '/Users/max/Documents/_Uni/DTU/MLOPS/cnn_mnist/data/processed/test',
                                     'lr': 0.001,
                                     'batch_size': 64,
                                     'epochs': 1
                                     }
                                }
                           }
                      }
                     )

os.environ['WANDB_SILENT'] = "true"


class TestTraining(TestCase):
    def test_training(self):
        with self.assertLogs() as captured:
            train(config)
        a = captured.records
        self.assertEqual(captured.records[1].getMessage(), "Start training..."), "Training is not started"
        self.assertEqual(captured.records[-1].getMessage(), "Finished training and saved model!")
