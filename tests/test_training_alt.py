# """https://coderbook.com/@marcus/how-to-unit-test-functions-without-return-statements-in-python/"""
# from unittest.mock import Mock, patch
# from src.models.train_model import train
# from hydra import initialize, compose
# from omegaconf import OmegaConf
# import os
#
#
# config = OmegaConf.create({'experiment':
#                           {'exp1':
#                                {'model':
#                                     {'image_width': 28, 'image_height': 28,
#                                      'con_layer1':
#                                          {'input_channel': 1, 'output_channel': 4, 'kernel_size': 3, 'padding': 1,
#                                           'stride': 1},
#                                      'max_pool1': {'kernel_size': 2, 'stride': 2},
#                                      'con_layer2': {'input_channel': 1, 'output_channel': 4, 'kernel_size': 3,
#                                                     'padding': 1, 'stride': 1},
#                                      'max_pool2': {'kernel_stride': 2, 'stride': 2}, 'output_dim': 10},
#                                 'training':
#                                     {'data_path': '/Users/max/Documents/_Uni/DTU/MLOPS/cnn_mnist/data/processed/test',
#                                      'lr': 0.001,
#                                      'batch_size': 64,
#                                      'epochs': 1
#                                      }
#                                 }
#                            }
#                       }
#                      )
#
#
# os.environ['WANDB_SILENT'] = "true"
#
#
# @patch("src.models.model.MyAwesomeModel")
# def test_training(mock_bar):
#     train(config)
#     # mock_bar.assert_called_once()
#     a = mock_bar.call_args_list
#     b = 1
