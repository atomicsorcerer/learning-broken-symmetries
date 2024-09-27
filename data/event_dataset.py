import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from matplotlib import pyplot as plt


class EventDataset(Dataset):
	def __init__(
			self,
			bg_file_path: str,
			signal_file_path: str,
			feature_cols: list[str],
			features_shape: tuple,
			limit: int = 10_000,
			shuffle_seed: int | None = None,
			blur_size: float = 0.0,
			n_bins: int = 100
	) -> None:
		"""
		Initializes an EventDataset for given CSV files of signal and background.

		Args:
			bg_file_path: File path to the CSV file with data on the background events.
			signal_file_path:File path to the CSV file with data on the signal events.
			limit: Optional limit on number of events to sample for the dataset.
			shuffle_seed: Optional shuffle seed for reproducibility.
		"""
		if shuffle_seed is None:
			shuffle_seed = np.random.randint(0, 100)
		
		bg_dataset = pl.read_csv(bg_file_path).with_columns(
			pl.Series([0]).alias("label")
		)
		signal_dataset = pl.read_csv(signal_file_path).with_columns(
			pl.Series([1]).alias("label")
		)
		
		amalgam_dataset = pl.concat((bg_dataset, signal_dataset))
		amalgam_dataset = amalgam_dataset.with_columns(
			[(np.sqrt(pl.col("px_0") ** 2 + pl.col("py_0") ** 2)).alias("pT_0"),
			 (np.sqrt(pl.col("px_1") ** 2 + pl.col("py_1") ** 2)).alias("pT_1"),
			 (np.sqrt(
				 (pl.col("energy_0") + pl.col("energy_1")) ** 2
				 - (pl.col("px_0") + pl.col("px_1")) ** 2
				 - (pl.col("py_0") + pl.col("py_1")) ** 2
				 - (pl.col("pz_0") + pl.col("pz_1")) ** 2
			 )).alias("muon_pair_inv_mass")])
		
		blur = np.random.default_rng(shuffle_seed).normal(0,
		                                                  amalgam_dataset.get_column("pT_0").to_numpy() * blur_size)
		amalgam_dataset = amalgam_dataset.with_columns([
			(np.arctan(pl.col("py_0") / pl.col("px_0"))).alias("phi_0"),
			(np.arctan(pl.col("py_1") / pl.col("px_1"))).alias("phi_1")
		])
		
		amalgam_dataset = amalgam_dataset.with_columns([
			(pl.col("pT_0") + blur).alias("blurred_pT_0"),
			(pl.col("pT_1") + blur).alias("blurred_pT_1")
		])
		amalgam_dataset = amalgam_dataset.with_columns([
			(np.sign(pl.col("px_0")) * np.abs(pl.col("blurred_pT_0") * np.cos(pl.col("phi_0")))).alias(
				"blurred_px_0"),
			(np.sign(pl.col("px_1")) * np.abs(pl.col("blurred_pT_1") * np.cos(pl.col("phi_1")))).alias(
				"blurred_px_1"),
			
			(np.sign(pl.col("py_0")) * np.abs(pl.col("blurred_pT_0") * np.sin(pl.col("phi_0")))).alias(
				"blurred_py_0"),
			(np.sign(pl.col("py_1")) * np.abs(pl.col("blurred_pT_1") * np.sin(pl.col("phi_1")))).alias(
				"blurred_py_1")
		])
		amalgam_dataset = amalgam_dataset.with_columns([
			(np.sqrt(
				pl.col("blurred_px_0") ** 2 - pl.col("px_0") ** 2
				+ pl.col("blurred_py_0") ** 2 - pl.col("py_0") ** 2
				+ pl.col("energy_0") ** 2
			)).alias("blurred_energy_0"),
			(np.sqrt(
				pl.col("blurred_px_1") ** 2 - pl.col("px_1") ** 2
				+ pl.col("blurred_py_1") ** 2 - pl.col("py_1") ** 2
				+ pl.col("energy_1") ** 2
			)).alias("blurred_energy_1"),
		])
		amalgam_dataset = amalgam_dataset.with_columns(
			(np.sqrt(
				(pl.col("blurred_energy_0") + pl.col("blurred_energy_1")) ** 2
				- (pl.col("blurred_px_0") + pl.col("blurred_px_1")) ** 2
				- (pl.col("blurred_py_0") + pl.col("blurred_py_1")) ** 2
				- (pl.col("pz_0") + pl.col("pz_1")) ** 2
			)).alias("blurred_muon_pair_inv_mass"))
		
		bg_distro, x_axis = np.histogram(
			amalgam_dataset.filter(
				pl.col("label") == 0
			).get_column("blurred_pT_0").to_numpy(), n_bins)
		
		signal_distro, _ = np.histogram(
			amalgam_dataset.filter(
				pl.col("label") == 1
			).get_column("blurred_pT_0").to_numpy(), n_bins, range=(min(x_axis), max(x_axis))
		)
		
		x_axis = x_axis[:-1]
		
		pT_indices = np.searchsorted(x_axis, amalgam_dataset.get_column("blurred_pT_0").to_numpy(),
		                             side="right") - 1
		amalgam_dataset = amalgam_dataset.with_columns([
			pl.Series("signal_distro_weight", signal_distro[pT_indices]),
			pl.Series("bg_distro_weight", bg_distro[pT_indices])
		])
		
		amalgam_dataset = amalgam_dataset.with_columns(
			(pl.col("signal_distro_weight") / (
					pl.col("label") * pl.col("signal_distro_weight")
					+ (1 - pl.col("label")) * pl.col("bg_distro_weight")))
			.fill_nan(
				0.0)  # NaN values are caused when both background and signal have no events on the wanted range
			.replace(np.inf, 0.0)  # Infinite values indicate that the background event is outside the distribution
			.alias("norm_weight")
		)
		
		amalgam_dataset = amalgam_dataset.sample(
			min(limit, len(amalgam_dataset)),
			shuffle=True if (shuffle_seed is not None) else False,
			seed=shuffle_seed,
		)
		
		norm_weights = amalgam_dataset.get_column("norm_weight")
		self.norm_weights = torch.Tensor(norm_weights)
		
		amalgam_dataset = amalgam_dataset.select([*feature_cols, "label"])
		
		labels = amalgam_dataset.get_column("label").to_list()
		labels = np.array(labels, dtype=np.float32)
		self.labels = torch.Tensor(labels)
		
		features = amalgam_dataset.drop("label").to_numpy().reshape(features_shape).tolist()
		features = torch.Tensor(features).type(torch.float32)
		
		self.features = features
		self.features_shape = features_shape
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
		return self.features[idx], torch.unsqueeze(self.labels[idx], 0)


if __name__ == "__main__":
	blur_size = 0.10
	feature_cols = [
		"blurred_muon_pair_inv_mass", "blurred_pT_0"
	]
	data = EventDataset("background.csv",
	                    "signal.csv",
	                    feature_cols,
	                    features_shape=(-1, 2),
	                    limit=20_000,
	                    blur_size=blur_size,
	                    shuffle_seed=314,
	                    n_bins=100)
	
	sampler = WeightedRandomSampler(data.norm_weights, len(data), replacement=True,
	                                generator=torch.Generator().manual_seed(314))
	
	classes, label = data[list(sampler)]
	label = label.squeeze(0)
	
	# Mass for different slices of pT
	
	figure, axis = plt.subplots(3, sharex=True, sharey=True)
	figure.suptitle(f"Signal muon pair invariant mass for different slices of pT (blur = {blur_size * 100}%)")
	figure.supxlabel("Mass")
	figure.supylabel("Entries")
	
	limit = (0.0, max(classes[label == 0][..., 0].numpy().reshape(-1)))
	axis[0].hist(classes[(classes[..., 1] < 30) & (label == 1)][..., 0], bins=100, range=limit)
	axis[0].set_title("pT < 30")
	
	axis[1].hist(classes[(30 < classes[..., 1]) & (classes[..., 1] < 40) & (label == 1)][..., 0], bins=100, range=limit)
	axis[1].set_title("30 < pT < 40")
	
	axis[2].hist(classes[(classes[..., 1] > 40) & (label == 1)][..., 0], bins=100, range=limit)
	axis[2].set_title("pT > 40")
	
	figure.set_size_inches(12, 8)
	plt.savefig("../figures/signal mass distribution (pT slices).pdf")
	plt.show()
	
	exit()
	
	# pT distribution before and after reweighting
	
	figure, axis = plt.subplots(2, 2, sharex=True, sharey=True)
	figure.suptitle(
		f"pT distributions for signal and background, before and after reweighting (blur = {blur_size * 100}%)")
	figure.supxlabel("pT")
	figure.supylabel("Entries")
	
	bins = 100
	limit = [
		float(min(data.features[..., 1])), float(max(data.features[..., 1]))
	]
	axis[0][0].hist(data.features[data.labels == 1][..., 1].numpy().reshape(-1), bins=bins, range=limit,
	                color="tab:orange")
	axis[0][0].set_title("Signal distribution (original)")
	
	axis[0][1].hist(data.features[data.labels == 0][..., 1].numpy().reshape(-1), bins=bins, range=limit,
	                color="tab:blue")
	axis[0][1].set_title("Background distribution (original)")
	
	axis[1][0].hist(classes[label == 1][..., 1].numpy().reshape(-1), bins=bins, range=limit, color="tab:orange")
	axis[1][0].set_title("Signal distribution (with reweight)")
	
	axis[1][1].hist(classes[label == 0][..., 1].numpy().reshape(-1), bins=bins, range=limit, color="tab:blue")
	axis[1][1].set_title("Background distribution (with reweight)")
	
	figure.set_size_inches(12, 8)
	plt.savefig("../figures/pT distributions.pdf", dpi=600)
	plt.show()
	
	figure, axis = plt.subplots(2, 2, sharex=True, sharey=True)
	figure.suptitle(
		f"Mass distributions for signal and background, before and after reweighting (blur = {blur_size * 100}%)")
	figure.supxlabel("Mass")
	figure.supylabel("Entries")
	
	limit = [
		float(min(data.features[..., 0])), float(max(data.features[..., 0]))
	]
	axis[0][0].hist(data.features[data.labels == 1][..., 0].numpy().reshape(-1), bins=bins, range=limit,
	                color="tab:orange")
	axis[0][0].set_title("Signal distribution (original)")
	
	axis[0][1].hist(data.features[data.labels == 0][..., 0].numpy().reshape(-1), bins=bins, range=limit,
	                color="tab:blue")
	axis[0][1].set_title("Background distribution (original)")
	
	axis[1][0].hist(classes[label == 1][..., 0].numpy().reshape(-1), bins=bins, range=limit, color="tab:orange")
	axis[1][0].set_title("Signal distribution (with reweight)")
	
	axis[1][1].hist(classes[label == 0][..., 0].numpy().reshape(-1), bins=bins, range=limit, color="tab:blue")
	axis[1][1].set_title("Background distribution (with reweight)")
	
	figure.set_size_inches(12, 8)
	plt.savefig("../figures/mass distributions.pdf", dpi=600)
	plt.show()
