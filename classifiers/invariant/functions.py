import torch


def minkowski_inner_product(p: torch.Tensor, q: torch.Tensor, c: int = 1) -> torch.Tensor:
	"""
	Computes the Minkowski inner product between the four-momenta of two particles.
	
	Args:
		p: Tensor of four-momentum of first particle (x, y, z, t).
		q: Tensor of four-momentum of second particle (x, y, z, t).
		c: Speed of light constant.

	Returns:
		torch.Tensor: The Minkowski inner product of the two vectors.
	"""
	return c ** 2 * p[3] * q[3] - p[2] * q[2] - p[1] * q[1] - p[0] * q[0]


def squared_minkowski_norm(p: torch.Tensor) -> torch.Tensor:
	"""
	Computes the square of the Minkowski norm of a vector.
	
	Args:
		p: Tensor of four-momentum of a particle (x, y, z, t).

	Returns:
		torch.Tensor: The square of the Minkowski norm of the vector.
	"""
	p_squared = torch.pow(p, 2)
	
	return p_squared[3] - p_squared[0] - p_squared[1] - p_squared[2]
