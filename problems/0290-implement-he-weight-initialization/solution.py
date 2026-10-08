import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    np.random.seed(seed)

    fan = n_in if mode == 'fan_in' else n_out

    if distribution == 'normal':
        std = (2.0 / fan) ** 0.5
        weights = np.random.normal(0, std, size=(n_in, n_out))   
        #(n_in, n_out) * std
    else:
        bound = (6.0 / fan) ** 0.5
        weights = np.random.uniform(- bound, bound, size=(n_in, n_out))
        #torch.rand(n_in, n_out) * 2 * bound - bound

    return weights