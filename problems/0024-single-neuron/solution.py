import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	z = np.dot(features, weights) + bias

	sigma = 1/(1 + np.exp(-z))

	mse = np.mean((sigma - labels) ** 2)

	return sigma, mse