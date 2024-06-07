import torch


def train(dataloader, model, loss_fn, optimizer) -> None:
	"""
	Train a model.

	Args:
		dataloader: Data to be used in training.
		model: Model to be trained.
		loss_fn: Loss function to be used.
		optimizer: Optimizer to be used.
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
		
		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
			print(f"loss: {loss:>5f}\t [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn) -> None:
	"""
	Test a model.

	Args:
		dataloader: Data to be used in training.
		model: Model to be trained.
		loss_fn: Loss function to be used.
	"""
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			
			for i_y, i_pred in zip(list(y), list(pred)):
				i_y = i_y.numpy()
				i_pred = torch.round(torch.nn.functional.sigmoid(i_pred)).numpy()
				correct += 1 if i_y == i_pred else 0
	
	test_loss /= num_batches
	
	print(f"Test Error: Avg loss: {test_loss:>8f}")
	print(f"Accuracy: {correct}/{size:>0.1f} = {correct / size * 100:<0.2f}% \n")
