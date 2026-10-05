import math

def sigmoid(z: float) -> float:
	return 1/ (1+ math.exp(-z))