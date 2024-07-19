import torch


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
		pred = model(X)
		loss = loss_fn(pred, y)
		
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
			
			auc_input = torch.cat((auc_input, y.reshape((-1))))
			auc_target = torch.cat((auc_target, torch.nn.functional.sigmoid(pred).reshape((-1))))
			
			for i_y, i_pred in zip(list(y), list(pred)):
				i_y = i_y.numpy()
				i_pred = torch.round(torch.nn.functional.sigmoid(i_pred)).numpy()
				correct += 1 if i_y == i_pred else 0
	
	test_loss /= num_batches
	
	metric.update(auc_input, auc_target)
	auc = metric.compute().item()
	
	if print_results:
		print(f"Test Error: Avg loss: {test_loss:>8f}")
		print(f"Accuracy: {correct}/{size:>0.1f} = {correct / size * 100:<0.2f}% \n")
		print(f"AUC: {auc:>0.1f} \n")
	
	return test_loss, correct / size, auc
