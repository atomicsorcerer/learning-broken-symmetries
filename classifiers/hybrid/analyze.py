import torch
from torch.utils.data import WeightedRandomSampler

from matplotlib import pyplot as plt
import polars as pl

from data import EventDataset

blur_size = 0.10
feature_cols = [
	"blurred_pT_0"
]
data = EventDataset("../../data/background.csv",
                    "../../data/signal.csv",
                    feature_cols,
                    features_shape=(-1, 1),
                    limit=20_000,
                    blur_size=blur_size,
                    shuffle_seed=314)

sampler = WeightedRandomSampler(data.norm_weights, len(data), replacement=True,
                                generator=torch.Generator().manual_seed(314))
dataset = data[list(sampler)][0]

pfn_model = torch.load("model.pth")

# Analysis of weight identification function

general_weight_result = pfn_model.weight_network(dataset)

figure = plt.figure(1, figsize=(12, 8), dpi=600)
figure.suptitle("Proportion of general subnet used in hybrid classifier vs. pT")
plt.scatter(list(dataset), list(general_weight_result[..., 0].detach().numpy()), color="tab:blue")
plt.xlabel("pT")
plt.ylabel("Proportion of general subnet used in hybrid classifier")
figure.savefig("../../figures/general subnet proportion vs pT.pdf")
plt.show()

pT_vs_gen_prop_log = pl.DataFrame({
	"pT": list(dataset),
	"general classifier proportion": list(general_weight_result[..., 0].detach().numpy()),
	"label": list(data[list(sampler)][1].flatten()),
	"classifier": "Latent Space Pooled Hybrid Classifier",
	"blur": blur_size
})
pT_vs_gen_prop_log.write_csv("weight_subnet_analysis.csv")

# Analysis of the accuracy/confidence of the classifier vs. pT

blur_size = 0.10
feature_cols = [
	"blurred_px_0", "blurred_py_0", "pz_0", "blurred_energy_0", "blurred_px_1", "blurred_py_1", "pz_1",
	"blurred_energy_1"
]
data = EventDataset("../../data/background.csv",
                    "../../data/signal.csv",
                    feature_cols,
                    features_shape=(-1, 2, 4),
                    limit=20_000,
                    blur_size=blur_size,
                    shuffle_seed=314)

sampler = WeightedRandomSampler(data.norm_weights, len(data), replacement=True,
                                generator=torch.Generator().manual_seed(314))
dataset = data[list(sampler)]
X = dataset[0]
Y = dataset[1].squeeze().unsqueeze(1)

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
plt.title("Latent Space Pooled Hybrid Classifier - pT vs. accuracy")
plt.show()

pT_vs_acc_log = pl.DataFrame({
	"pT": list(map(lambda x: x[0], coords)),
	"muon_inv_mass": list(map(lambda x: x[4], coords)),
	"acc": list(map(lambda x: x[1], coords)),
	"output": list(map(lambda x: x[2].item(), coords)),
	"correct_output": list(map(lambda x: x[3].item(), coords)),
	"classifier": "Latent Space Pooled Hybrid Classifier",
	"blur": blur_size
})
pT_vs_acc_log.write_csv("pT_vs_acc_analysis.csv")
