import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader


ROOT_PATH = Path(__file__).resolve().parents[2]


def train() -> None:
    """
    Saves trained CNN model

        :parameter:
        :returns:

    """
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--data-path", help="Subcommand to run")
    args = parser.parse_args()
    data_path = args.data_path

    # hyperparameter definition
    model = MyAwesomeModel()

    train_set = torch.load(data_path)
    criterion = nn.NLLLoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 5
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    acc_epoch = []
    loss_epoch = []
    for e in range(epochs):
        running_loss = 0
        acc = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            img, labels = data
            x = img.unsqueeze(1)
            output = model(x.float())
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc.append(torch.mean(equals.type(torch.FloatTensor)).item())
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        else:
            print(
                f"Epoch {e + 1} --> Loss: {running_loss:.6} "
                f"Accuracy: {np.sum(acc) / len(acc) * 100:.4}%"
            )
            acc_epoch.append(np.sum(acc) / len(acc) * 100)
            loss_epoch.append(running_loss)

    i = 0
    model_path = os.path.join(ROOT_PATH, "models")
    while True:
        if not os.path.isfile(
            os.path.join(model_path, f"model_e{epochs}_b{batch_size}_lr{lr}_{i}.pth")
        ):
            break
        else:
            i += 1
    torch.save(
        model.state_dict(),
        os.path.join(model_path, f"model_e{epochs}_b{batch_size}_lr{lr}_{i}.pth"),
    )

    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.plot(acc_epoch), plt.xlabel("epochs"), plt.ylabel("accuracy")
    plt.subplot(122), plt.plot(loss_epoch), plt.xlabel("epochs"), plt.ylabel("loss")
    plt.savefig(
        os.path.join(ROOT_PATH, f"reports/figures/model_e{epochs}_b{batch_size}_lr{lr}_{i}.png")
    )
    plt.show()


if __name__ == "__main__":
    train()
