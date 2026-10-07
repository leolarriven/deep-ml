import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	first_part = np.linalg.inv(np.transpose(X) @ X)

	second_part = np.transpose(X) @ y

	return first_part @ second_part