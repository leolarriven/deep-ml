import torch

def grad_of_quadratic(x_value: float) -> float:
    
    x = torch.tensor(x_value, requires_grad = True)

    function = x**2 + 3*x +2

    function.backward()

    return x.grad.item()