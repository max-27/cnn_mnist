import wandb
import os
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf.dictconfig import DictConfig
from src.models.model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader
from src import _PATH_DATA


ROOT_PATH = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config: DictConfig) -> None:
    """
    Saves trained CNN model

        :parameter:
        :returns:
    """
    if os.getcwd().split("/")[-1] == "tests":
        mode_wandb = "disabled"
    else:
        mode_wandb = "online"
    wandb.init(
        project="test_project",
        entity="yeah_42",
        name=os.getcwd().split('/')[-1],
        job_type="train",
        mode=mode_wandb,
        force=True,
    )
    wandb.config = config
    logger.info(f"Experiment setting: {config.experiment.items()[0][0]}")
    cfg_exp = config.experiment.items()[0][1]
    # cfg_model = cfg_exp.model
    cfg_train = cfg_exp.training

    # hyperparameter definition
    model = MyAwesomeModel()
    wandb.watch(model, log_freq=100)

    train_set = torch.load(os.path.join(_PATH_DATA, cfg_train.data_path))
    criterion = nn.NLLLoss()
    lr = cfg_train.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = cfg_train.epochs
    batch_size = cfg_train.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    acc_epoch = []
    loss_epoch = []
    logger.info("Start training...")
    for e in range(epochs):
        running_loss = 0
        acc = []
        sample_img = None
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            img, labels = data
            sample_img = img[-1]
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
            logger.info(
                f"Epoch {e + 1} --> Loss: {running_loss:.6} "
                f"Accuracy: {np.sum(acc) / len(acc) * 100:.4}%"
            )
            wandb.log(
                {"training_loss": running_loss,
                 "training_acc": np.sum(acc) / len(acc) * 100}
            )
            images = wandb.Image(
                sample_img, f"Last image in epoch {e}"
            )
            wandb.log(
                {"Sample images": images}
            )
            acc_epoch.append(np.sum(acc) / len(acc) * 100)
            loss_epoch.append(running_loss)

    if os.getcwd().split("/")[-1] != "tests":  # don't save during tests
        torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # plt.figure(figsize=(15, 10))
    # plt.subplot(121), plt.plot(acc_epoch), plt.xlabel("epochs"), plt.ylabel("accuracy")
    # plt.subplot(122), plt.plot(loss_epoch), plt.xlabel("epochs"), plt.ylabel("loss")
    # plot_path = os.path.join(ROOT_PATH, "reports/figures/")
    # os.makedirs(plot_path, exist_ok=True)
    # plt.savefig(
    #     os.path.join(plot_path, f"{os.getcwd()}/learning_curves.png")
    # )
    # plt.show()
    wandb.finish()
    logger.info("Finished training and saved model!")


if __name__ == "__main__":
    train()
