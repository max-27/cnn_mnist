import torch
import torchdrift
from torch.utils.data import DataLoader
from src.models.model import MyAwesomeModel
import matplotlib.pyplot as plt


def corruption_function(x: torch.Tensor):
    x = x.float()
    x = torchdrift.data.functional.gaussian_blur(x, severity=2)
    return x

def collate_fn(batch):
    batch = torch.utils.data._utils.collate.default_collate(batch)
    batch = (corruption_function(batch[0].unsqueeze(1)), *batch[1:])

test_dataset = torch.load("/Users/max/Documents/_Uni/DTU/MLOPS/cnn_mnist/data/processed/test")

test_loader_corr = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=16)
test_loader_ucorr = DataLoader(test_dataset, batch_size=16)
test_inputs_ucorr, _ = next(iter(test_loader_ucorr))
test_inputs_corr, _ = next(iter(test_loader_corr))

model_weights = torch.load("/Users/max/Documents/_Uni/DTU/MLOPS/cnn_mnist/models/runs/2022-01-17_19-57-38/deploy_model.pth")
model = MyAwesomeModel()
model.load_state_dict(model_weights)

drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
torchdrift.utils.fit(test_dataloader(), model, drift_detector)

drift_detection_model = torch.nn.Sequential(
    model,
    drift_detector
)

features = feature_extractor(inputs)
score = drift_detector(features)
p_val = drift_detector.compute_p_value(features)
print(score, p_val)

# plot drift detector with normal input
N_base = drift_detector.base_outputs.size(0)
mapper = sklearn.manifold.Isomap(n_components=2)
base_embedded = mapper.fit_transform(drift_detector.base_outputs)
features_embedded = mapper.transform(features)
plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
plt.title(f'score {score:.2f} p-value {p_val:.2f}')


# plot drift detector with corrupted input
features = model(test_inputs_corr)
score = drift_detector(features)
p_val = drift_detector.compute_p_value(features)
features_embedded = mapper.transform(features)
pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
pyplot.title(f'score {score:.2f} p-value {p_val:.2f}')