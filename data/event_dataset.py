import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


class EventDataset(Dataset):
	def __init__(
			self,
			bg_file_path: str,
			signal_file_path: str,
			feature_cols: list[str],
			features_shape: tuple,
			limit: int = 10_000,
			shuffle_seed: int | None = None,
			blur_data: bool = False,
			blur_size: float = 0.1
	) -> None:
		"""
		Initializes an EventDataset for given CSV files of signal and background.

		Args:
			bg_file_path: File path to the CSV file with data on the background events.
			signal_file_path:File path to the CSV file with data on the signal events.
			limit: Optional limit on number of events to sample for the dataset.
			shuffle_seed: Optional shuffle seed for reproducibility.
		"""
		if feature_cols is None:
			feature_cols = ["px_0", "py_0", "pz_0", "energy_0", "px_1", "py_1", "pz_1", "energy_1"]
		if shuffle_seed is None:
			shuffle_seed = np.random.randint(0, 100)
		
		bg_dataset = pl.read_csv(bg_file_path).with_columns(
			pl.Series([0]).alias("label")
		)
		signal_dataset = pl.read_csv(signal_file_path).with_columns(
			pl.Series([1]).alias("label")
		)
		amalgam_dataset = pl.concat((bg_dataset, signal_dataset)).select([*feature_cols, "label"]).sample(
			limit,
			shuffle=True if shuffle_seed is not None else False,
			seed=shuffle_seed,
		)
		
		labels = amalgam_dataset.get_column("label").to_list()
		labels = np.array(labels, dtype=np.float32)
		self.labels = torch.Tensor(labels)
		
		features = amalgam_dataset.drop("label").to_numpy().reshape(features_shape).tolist()
		features = torch.Tensor(features).type(torch.float32)
		
		self.features = features
		self.features_shape = features_shape
		self.blur_data = blur_data
		self.blur_size = blur_size
	
	def __len__(self) -> int:
		"""
		Calculates the number of events in the dataset.

		Returns:
			int: Number of events in the dataset
		"""
		return len(self.labels)
	
	def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Gets the features and label for a given index in the dataset.

		Args:
			idx: Index of the feature to be returned.

		Returns:
			tuple: Feature and label at the index (feature, label)
		"""
		features = self.features[idx]
		
		if self.blur_data:
			temp_features = torch.Tensor(size=features.shape)
			
			pT = np.sqrt(features[..., 0] ** 2 + features[..., 1] ** 2)
			phi = torch.arctan(features[..., 1] / features[..., 0])
			
			blur = torch.normal(torch.zeros(len(pT)), pT * self.blur_size,
			                    generator=torch.Generator().manual_seed(31415))
			pT_new = pT + blur
			
			temp_features[..., 0] = torch.sign(features[..., 0]) * torch.abs(pT_new * torch.cos(phi))  # Recalculate px
			temp_features[..., 1] = torch.sign(features[..., 1]) * torch.abs(pT_new * torch.sin(phi))  # Recalculate py
			temp_features[..., 2] = features[..., 2]  # pz does not change
			temp_features[..., 3] = torch.sqrt(
				temp_features[..., 0] ** 2 - features[..., 0] ** 2
				+ temp_features[..., 1] ** 2 - features[..., 1] ** 2
				+ features[..., 3] ** 2
			)  # Recalculate E
			
			return temp_features, torch.unsqueeze(self.labels[idx], 0)
		
		return features, torch.unsqueeze(self.labels[idx], 0)


if __name__ == "__main__":
	cols = [
		"px_0", "py_0", "pz_0", "energy_0",
		"px_1", "py_1", "pz_1", "energy_1",
	]
	data = EventDataset("background.csv",
	                    "signal.csv",
	                    cols,
	                    features_shape=(-1, 2, 4),
	                    limit=20_000,
	                    blur_data=True,
	                    blur_size=0.10,
	                    shuffle_seed=314)
	
	dataloader = DataLoader(data, shuffle=True)
	
	X_signal = []
	Y_signal = []
	X_bg = []
	Y_bg = []
	for x in dataloader:
		if x[1].item() == 1:
			X_signal.append(x[0].tolist())
			Y_signal.append(x[1].item())
		else:
			X_bg.append(x[0].tolist())
			Y_bg.append(x[1].item())
	
	X_signal = torch.Tensor(X_signal)
	Y_signal = torch.Tensor(Y_signal).unsqueeze(1)
	
	pT_signal = torch.sqrt(
		torch.add(torch.pow(X_signal[..., 0][..., 0], 2), torch.pow(X_signal[..., 1][..., 0], 2))).flatten()
	pT_signal = pT_signal.tolist()
	muon_inv_mass_signal = torch.sqrt((X_signal[..., 3][..., 0] + X_signal[..., 3][..., 1]) ** 2
	                                  - ((X_signal[..., 0][..., 0] + X_signal[..., 0][..., 1]) ** 2
	                                     + (X_signal[..., 1][..., 0] + X_signal[..., 1][..., 1]) ** 2
	                                     + (X_signal[..., 2][..., 0] + X_signal[..., 2][..., 1]) ** 2)).flatten()
	
	X_bg = torch.Tensor(X_bg)
	Y_bg = torch.Tensor(Y_bg).unsqueeze(1)
	
	pT_bg = torch.sqrt(torch.add(torch.pow(X_bg[..., 0][..., 0], 2), torch.pow(X_bg[..., 1][..., 0], 2))).flatten()
	pT_bg = pT_bg.tolist()
	muon_inv_mass_bg = torch.sqrt((X_bg[..., 3][..., 0] + X_bg[..., 3][..., 1]) ** 2
	                              - ((X_bg[..., 0][..., 0] + X_bg[..., 0][..., 1]) ** 2
	                                 + (X_bg[..., 1][..., 0] + X_bg[..., 1][..., 1]) ** 2
	                                 + (X_bg[..., 2][..., 0] + X_bg[..., 2][..., 1]) ** 2)).flatten()
	
	n_bins = 50
	(signal_distro, x_axis, y_axis) = np.histogram2d(muon_inv_mass_signal, pT_signal, n_bins, density=False)
	signal_distro = signal_distro.reshape((-1,))
	signal_distro = signal_distro / signal_distro.sum()
	signal_distro = signal_distro.reshape((n_bins, n_bins))
	
	print("\t".join(map(str, list(x_axis)[:-1])))
	print("\t".join(map(str, list(y_axis)[:-1])))
	distro = pl.DataFrame(signal_distro)
	distro.write_csv("signal_distro.csv")
	
	axes = pl.DataFrame([list(x_axis)[:-1], list(y_axis)[:-1]], schema=["mass", "pT"])
	axes.write_csv("distro_axes.csv")
	
	fig, axs = plt.subplots(2)
	fig.suptitle("Distribution of pT and mass")
	
	axs[0].hist2d(pT_signal, muon_inv_mass_signal, bins=n_bins)
	axs[1].hist2d(pT_bg, muon_inv_mass_bg, bins=n_bins)
	
	axs[0].set_title("Signal")
	axs[1].set_title("Background")
	
	axs[0].set_xlabel("pT")
	axs[1].set_xlabel("pT")
	axs[0].set_ylabel("Mass")
	axs[1].set_ylabel("Mass")
	
	plt.show()
