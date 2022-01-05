"""https://learnopencv.com/t-sne-for-feature-visualization/"""
import os
import random
import argparse
import matplotlib.pyplot as plt

import torch
from sklearn.manifold import TSNE
import numpy as np

from src.models.model import MyAwesomeModel


def visualize():
    parser = argparse.ArgumentParser(description="Add paths")
    parser.add_argument("--model-path", help="Add model path")
    parser.add_argument("--data-path", help="Add data path")
    args = parser.parse_args()
    model = MyAwesomeModel()
    weights = torch.load(args.model_path)
    model.load_state_dict(weights)

    # TODO extract features from model
    features0 = model.cnn_layers[0].weight.data.squeeze()
    features1 = model.cnn_layers[4].weight.data.squeeze().flatten()
    features2 = model.linear_layers[0].weight.data

    image_set = torch.load(args.data_path)
    image_loader = torch.utils.data.DataLoader(image_set, batch_size=1, shuffle=True)
    output = model.forward_visualize(next(iter(image_loader))[0].unsqueeze(1).float())
    output = output.view(int(np.sqrt(len(output))), -1)


    # fix seed
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    features_embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(
        features2
    )
    x_scaled = scale_to_01_range(features_embedded[:, 0])
    y_scaled = scale_to_01_range(features_embedded[:, 1])

    plt.imshow(output.detach().numpy(), cmap="gray")
    plt.show()


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range


if __name__ == "__main__":
    visualize()
