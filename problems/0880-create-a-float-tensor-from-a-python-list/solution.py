import torch

def to_float_tensor(values):
    return torch.tensor(values, dtype=torch.float32)
