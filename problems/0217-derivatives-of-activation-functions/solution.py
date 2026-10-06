import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# derivate of sigmoid
	sigmoid_function = 1 / (1 + np.exp(-x))
	d_sigmoid = sigmoid_function * (1 - sigmoid_function)

	# derviate of hyperbolic tangent
	tanh_function = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	d_tanh = 1 - tanh_function ** 2

	# derivate of relu
	d_relu = 1.0 if x > 0 else 0.0

	dic = {'sigmoid': d_sigmoid, 'tanh': d_tanh, 'relu': d_relu}
	return dic