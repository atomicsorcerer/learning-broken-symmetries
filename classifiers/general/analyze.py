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

pfn_model = torch.load("model.pth")

pT = []
inv_mass = []
acc = []
output = []
correct_output = []

for X in data:
	y = X[1].item()
	correct_output.append(y)
	
	muon_inv_mass = torch.sqrt((X[0][0][3] + X[0][1][3]) ** 2
	                           - ((X[0][0][0] + X[0][1][0]) ** 2
	                              + (X[0][0][1] + X[0][1][1]) ** 2
	                              + (X[0][0][2] + X[0][1][2]) ** 2))
	inv_mass.append(muon_inv_mass.item())
	
	pT_x = torch.sqrt(torch.add(torch.pow(X[0][0][0], 2), torch.pow(X[0][0][1], 2)))
	X = X[0]
	
	result = pfn_model(X.unsqueeze(0))
	result = torch.nn.functional.sigmoid(result).item()
	output.append(result)
	result = 1 - np.abs((result - y))
	
	pT.append(pT_x)
	acc.append(result)

coords = sorted(zip(pT, inv_mass, acc, output, correct_output), key=lambda x: x[0])

plt.plot(list(map(lambda x: x[0], coords)), list(map(lambda x: x[2], coords)), marker="o", linestyle="", markersize=0.5)
plt.xlabel("pT")
plt.ylabel("Accuracy")
plt.title("PFN Classifier - pT vs. accuracy")
plt.show()

log = pl.DataFrame({
	"pT": list(map(lambda x: x[0], coords)),
	"muon_inv_mass": list(map(lambda x: x[1], coords)),
	"acc": list(map(lambda x: x[2], coords)),
	"output": list(map(lambda x: x[3], coords)),
	"correct_output": list(map(lambda x: x[4], coords)),
	"classifier": "PFN Classifier",
	"blur": blur_size
})
log.write_csv("analysis.csv")
