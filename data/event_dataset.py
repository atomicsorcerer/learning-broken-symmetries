import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class EventDataset(Dataset):
	def __init__(
			self,
			bg_file_path: str,
			signal_file_path: str,
			feature_cols: list[str],
			features_shape: tuple,
			limit: int = 10_000,
			shuffle_seed=None,
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
		features = torch.Tensor(features)
		
		self.features = features
	
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
