import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    # Initialisation
    
    hidden_state = torch.tensor(initial_hidden_state)
    Wx = torch.tensor(Wx)
    Wh = torch.tensor(Wh)
    b = torch.tensor(b)

    # Sequence Processing

    for val in input_sequence:
        val = torch.tensor(val)
        hidden_state = torch.tanh(Wx @ val + Wh @ hidden_state + b)
    
    return hidden_state


