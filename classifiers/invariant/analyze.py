import torch
from torch.utils.data import random_split

import numpy as np
from matplotlib import pyplot as plt
import polars as pl

from data import EventDataset

blur_size = 0.10
feature_cols = ["px_0", "py_0", "pz_0", "energy_0", "px_1", "py_1", "pz_1", "energy_1"]
data = EventDataset("../../data/background.csv",
                    "../../data/signal.csv",
                    feature_cols,
                    features_shape=(-1, 2, 4),
                    limit=20_000,
                    blur_data=True,
                    blur_size=blur_size,
                    shuffle_seed=314)

test_percent = 0.20
_, test_data = random_split(data, [1 - test_percent, test_percent], torch.Generator().manual_seed(314))
data = list(test_data)

X = []
Y = []
for x in test_data:
	X.append(x[0].tolist())
	Y.append(x[1].item())

X = torch.Tensor(X)
Y = torch.Tensor(Y).unsqueeze(1)

pfn_model = torch.load("model.pth")
result = pfn_model(X)
result = torch.nn.functional.sigmoid(result)
acc = 1 - torch.abs(torch.subtract(result, Y)).squeeze()
acc = acc.tolist()
pT = torch.sqrt(torch.add(torch.pow(X[..., 0][..., 0], 2), torch.pow(X[..., 1][..., 0], 2)))
pT = pT.tolist()
muon_inv_mass = torch.sqrt((X[..., 3][..., 0] + X[..., 3][..., 1]) ** 2
                           - ((X[..., 0][..., 0] + X[..., 0][..., 1]) ** 2
                              + (X[..., 1][..., 0] + X[..., 1][..., 1]) ** 2
                              + (X[..., 2][..., 0] + X[..., 2][..., 1]) ** 2))

coords = sorted(zip(pT, acc, result, Y, muon_inv_mass), key=lambda x: x[0])

plt.plot(list(map(lambda x: x[0], coords)), list(map(lambda x: x[1], coords)), marker="o", linestyle="", markersize=0.5)
plt.xlabel("pT")
plt.ylabel("Accuracy")
plt.title("Lorenz-invariant Classifier - pT vs. accuracy")
plt.show()

log = pl.DataFrame({
	"pT": list(map(lambda x: x[0], coords)),
	"muon_inv_mass": list(map(lambda x: x[4], coords)),
	"acc": list(map(lambda x: x[1], coords)),
	"output": list(map(lambda x: x[2].item(), coords)),
	"correct_output": list(map(lambda x: x[3].item(), coords)),
	"classifier": "Lorentz-invairant Classifier",
	"blur": blur_size
})
log.write_csv("analysis.csv")
