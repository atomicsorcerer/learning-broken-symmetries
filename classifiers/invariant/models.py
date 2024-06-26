import torch
from torch import nn


class LorenzInvariantParticleMapping(nn.Module):
	def __init__(self, output_dimension: int, hidden_layer_dimensions=None) -> None:
		"""
		Maps each set of observables of a particle to a specific dimensional output using Lorenz-invariant functions.

		Args:
			output_dimension: The fixed number of output nodes.
			hidden_layer_dimensions: A list of numbers which set the sizes of hidden layers.

		Raises:
			TypeError: If hidden_layer_dimensions is not a list.
			ValueError: If hidden_layer_dimensions is an empty list.
		"""
		super().__init__()
		
		self.output_dimension = output_dimension
		
		if hidden_layer_dimensions is None:
			hidden_layer_dimensions = [100]
		elif not isinstance(hidden_layer_dimensions, list):
			raise TypeError(f"Hidden layer dimensions must be a valid list. {hidden_layer_dimensions} is not valid.")
		elif len(hidden_layer_dimensions) == 0:
			raise ValueError("Hidden layer dimensions cannot be empty.")
		
		stack = nn.Sequential(nn.Linear(2, hidden_layer_dimensions[0]),
		                      nn.BatchNorm1d(hidden_layer_dimensions[0]), nn.ReLU())
		
		for i in range(len(hidden_layer_dimensions)):
			stack.append(
				nn.Linear(hidden_layer_dimensions[i],
				          hidden_layer_dimensions[i] if i == len(hidden_layer_dimensions) - 1 else
				          hidden_layer_dimensions[
					          i + 1]))
			stack.append(nn.BatchNorm1d(
				hidden_layer_dimensions[i] if i == len(hidden_layer_dimensions) - 1 else hidden_layer_dimensions[
					i + 1]))
			stack.append(nn.ReLU())
		
		stack.append(nn.Linear(hidden_layer_dimensions[-1], output_dimension))
		
		self.stack = stack
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward implementation for ParticleMapping.

		Args:
			x: Input tensor(s).

		Returns:
			torch.Tensor: Output tensor with predefined dimensions.
		"""
		minkowski_inner_product = (torch.mul(x[..., 3][..., 0], x[..., 3][..., 1])
		                           - torch.mul(x[..., 0][..., 0], x[..., 0][..., 1])
		                           - torch.mul(x[..., 1][..., 0], x[..., 1][..., 1])
		                           - torch.mul(x[..., 2][..., 0], x[..., 2][..., 1])).squeeze()
		
		squared_minkowski_norm = (torch.pow(torch.diff(x[..., 3]), 2)
		                          - torch.pow(torch.diff(x[..., 0]), 2)
		                          - torch.pow(torch.diff(x[..., 1]), 2)
		                          - torch.pow(torch.diff(x[..., 2]), 2)).squeeze()
		
		x = torch.stack([minkowski_inner_product, squared_minkowski_norm]).transpose(0, 1)
		x = self.stack(x)
		
		return x
