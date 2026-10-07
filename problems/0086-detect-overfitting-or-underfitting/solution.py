def model_fit_quality(training_accuracy, test_accuracy):
	"""
	Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
	:param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
	:param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
	:return: int, one of '1', '-1', or '0'.
	"""
	# Overfitting
	if training_accuracy - test_accuracy >= 0.20 :
		return 1
	
	elif training_accuracy < 0.70 and test_accuracy < 0.70 :
		return -1

	else :
		return 0
