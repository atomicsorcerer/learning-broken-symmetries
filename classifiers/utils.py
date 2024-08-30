import torch
import polars as pl


def train(dataloader, model, loss_fn, optimizer, print_results=False) -> None:
	"""
	Train a binary classification model.

	Args:
		dataloader: Data to be used in training.
		model: Model to be trained.
		loss_fn: Loss function to be used.
		optimizer: Optimizer to be used.
		print_results: Whether to print logs during training.
	"""
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction error
		pT = torch.sqrt(torch.add(torch.pow(X[..., 0][..., 0], 2), torch.pow(X[..., 1][..., 0], 2))).flatten()
		mass = torch.sqrt((X[..., 3][..., 0] + X[..., 3][..., 1]) ** 2
		                  - ((X[..., 0][..., 0] + X[..., 0][..., 1]) ** 2
		                     + (X[..., 1][..., 0] + X[..., 1][..., 1]) ** 2
		                     + (X[..., 2][..., 0] + X[..., 2][..., 1]) ** 2)).flatten()
		
		groupings = torch.Tensor(pl.read_csv("../../data/distro_axes.csv").to_numpy())
		distro = torch.Tensor(pl.read_csv("../../data/signal_distro.csv").to_numpy())
		weights = get_weights(mass, pT, distro, groupings[..., 0], groupings[..., 1])
		loss = torch.nn.BCEWithLogitsLoss(reduction="none")
		
		pred = model(X)
		loss = loss(pred, y)
		loss = torch.mean(loss * weights)
		
		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if batch % 100 == 0 and print_results:
			loss, current = loss.item(), (batch + 1) * len(X)
			print(f"loss: {loss:>5f}\t [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, metric, print_results=False) -> tuple[float, float, float]:
	"""
	Test a binary classification model.

	Args:
		dataloader: Data to be used in training.
		model: Model to be trained.
		loss_fn: Loss function to be used.
		metric: Metric to evaluate model performance.
		print_results: Whether to print logs during training.
	"""
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	
	test_loss, correct = 0, 0
	auc_input = torch.Tensor([])
	auc_target = torch.Tensor([])
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			
			auc_input = torch.cat((auc_input, torch.nn.functional.sigmoid(pred).reshape((-1))))
			auc_target = torch.cat((auc_target, y.reshape((-1))))
			
			for i_y, i_pred in zip(list(y), list(pred)):
				i_y = i_y.numpy()
				i_pred = torch.round(torch.nn.functional.sigmoid(i_pred)).numpy()
				correct += 1 if i_y == i_pred else 0
	
	test_loss /= num_batches
	
	metric.update(auc_input, auc_target)
	auc = metric.compute().item()
	
	if print_results:
		print(f"Test Error: Avg loss: {test_loss:>8f}")
		print(f"Accuracy: {correct}/{size:>0.1f} = {correct / size * 100:<0.2f}%")
		print(f"AUC: {auc:>0.3f} \n")
	
	return test_loss, correct / size, auc


def normalize(p: torch.Tensor) -> torch.Tensor:
	"""
	Normalizes large numbers to improve optimization.

	Adapted from the psi function from https://arxiv.org/pdf/2201.08187.

	Args:
		p: Input value to be normalized.

	Returns:
		torch.Tensor: Normalized output value.
	"""
	return torch.sign(p) * torch.log(torch.abs(p) + 1)


def get_weights(mass, pT, weight_matrix, mass_groupings, pT_groupings):
	mass_indices = torch.searchsorted(mass_groupings, mass, right=True) - 1
	pT_indices = torch.searchsorted(pT_groupings, pT, right=True) - 1
	
	weights = weight_matrix[pT_indices, mass_indices]
	
	return weights
