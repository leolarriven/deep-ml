import math

def softmax(scores: list[float]) -> list[float]:
    output_list = []

    # Numerical stability
    for i in scores:
        softmax_equation_denominator = 0
        for j in scores:
            softmax_equation_denominator += math.exp(j - max(scores))
        softmax_equation = math.exp(i - max(scores)) / softmax_equation_denominator
        output_list.append(softmax_equation)
    return output_list

