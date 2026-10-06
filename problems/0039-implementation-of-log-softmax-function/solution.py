import numpy as np

def log_softmax(scores: list) -> np.ndarray:

	right = 0

	for j in scores:
		right += np.exp(j-max(scores))
	
	total = []
	for i in scores:
		total.append(i - max(scores) - np.log(right))

	return total