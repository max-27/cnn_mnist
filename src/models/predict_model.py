import argparse
from pathlib import Path

import numpy as np
import torch
from model import MyAwesomeModel
from torch.utils.data import DataLoader

ROOT_PATH = Path(__file__).resolve().parents[2]


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Add relevant paths")
    parser.add_argument("--model-path", help="Add path to trained model")
    parser.add_argument("--data-path", help="Add path to test data")
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path

    model = MyAwesomeModel()
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    test_set = torch.load(data_path)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    with torch.no_grad():
        model.eval()
        acc = []
        for i, data in enumerate(test_loader):
            img, labels = data
            x = img.unsqueeze(1).float()
            output = model(x)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc.append(torch.mean(equals.type(torch.FloatTensor)).item())

        print(f"Final test accuracy: {np.sum(acc) / len(acc) * 100:.4}%")


if __name__ == "__main__":
    evaluate()
