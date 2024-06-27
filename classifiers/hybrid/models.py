import torch
from torch import nn

from classifiers.general import ParticleFlowNetwork
from classifiers.invariant import LorenzInvariantNetwork


class WeightedHybridClassifier(nn.Module):
	def __init__(self,
	             invariant_network_weight: float,
	             latent_space_dim: int,
	             pfn_mapping_hidden_layer_dimensions: list[int],
	             pfn_classifier_hidden_layer_dimensions: list[int],
	             lorenz_invariant_hidden_layer_dimensions: list[int]
	             ):
		"""
		Classifier that combines the result of two subnets using a weighted average.
		
		Args:
			invariant_network_weight: The weight on the invariant network result.
			latent_space_dim: Latent space dimension for the PFN general classifier.
			pfn_mapping_hidden_layer_dimensions: Hidden layers for the PFN mapping module.
			pfn_classifier_hidden_layer_dimensions: Hidden layers for the PFN general classifier.
			lorenz_invariant_hidden_layer_dimensions: Hidden layers for the Lorentz-invariant classifier.
		"""
		super().__init__()
		
		self.general_p_map = ParticleFlowNetwork(4,
		                                         8,
		                                         latent_space_dim,
		                                         pfn_classifier_hidden_layer_dimensions,
		                                         pfn_mapping_hidden_layer_dimensions)
		
		self.invariant_p_map = LorenzInvariantNetwork(1, lorenz_invariant_hidden_layer_dimensions)
		
		if invariant_network_weight > 1 or invariant_network_weight < 0:
			raise ValueError("Classifier weights must be between 0 and 1.")
		
		self.invariant_network_weight = invariant_network_weight
		self.general_network_weight = 1.0 - invariant_network_weight
	
	def forward(self, x):
		general_result = self.general_p_map(x)
		invariant_result = self.invariant_p_map(x)
		
		combined_result = (torch.mul(general_result, self.general_network_weight)
		                   + torch.mul(invariant_result, self.invariant_network_weight))
		
		return combined_result
