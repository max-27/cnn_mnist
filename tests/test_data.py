import os

import numpy as np
import pytest
import torch

from tests import _PATH_DATA, _PROJECT_ROOT


@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed")), reason="Data files not found")
def test_data():
    dataset_train = torch.load(os.path.join(_PROJECT_ROOT, "data/processed/train"))
    dataset_test = torch.load(os.path.join(_PROJECT_ROOT, "data/processed/test"))
    assert len(dataset_train) == 25000, f"Expected training set of length {25000} but got {len(dataset_train)} instead"
    assert len(dataset_test) == 5000, f"Expected training set of length {5000} but got {len(dataset_test)} instead"

    labels = []
    for data, label in dataset_train:
        assert data.shape == torch.Size([28, 28])
        assert type(label) == torch.Tensor
        labels.append(label)

    # assert all(i in labels for i in range(10))
    assert len(np.unique(labels)) == 10
