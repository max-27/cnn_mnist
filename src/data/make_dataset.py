# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
import torch.utils.data as data_utils
from torch import Tensor
from dotenv import find_dotenv, load_dotenv
from typing import List


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: Path, output_filepath: Path) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train_image_list = []
    train_label_list = []
    num_set = 8
    for i in range(num_set):
        img, labels = get_data(f"train_{i}.npz", input_filepath)
        train_image_list.append(img)
        train_label_list.append(labels)
    train_images = torch.as_tensor(train_image_list).view(num_set * 5000, 28, 28)
    train_labels = torch.as_tensor(train_label_list).view(num_set * 5000, -1).flatten()
    # TODO normalize tensor
    train = data_utils.TensorDataset(train_images, train_labels)

    test_image_list, test_label_list = get_data("test.npz", input_filepath)
    test_images = torch.as_tensor(test_image_list)
    test_labels = torch.as_tensor(test_label_list).flatten()
    test = data_utils.TensorDataset(test_images, test_labels)

    torch.save(train, os.path.join(output_filepath, "train"))
    torch.save(test, os.path.join(output_filepath, "test"))


def get_data(file_name: str, input_filepath: Path):
    data_file = os.path.join(input_filepath, file_name)
    data = np.load(data_file)
    return [data["images"], data["labels"]]


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
