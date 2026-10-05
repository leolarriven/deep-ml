import numpy as np


def moe(
    x: np.ndarray,
    We: np.ndarray,
    Wg: np.ndarray,
    n_experts: int,
    top_k: int
) -> np.ndarray:
    """
    Sparse Mixture-of-Experts layer.

    Args:
        x:  Input tensor, shape (B, S, D)
        We: Expert weights, shape (E, D, D)
        Wg: Gating weights, shape (D, E)
        n_experts: Number of experts
        top_k: Number of experts selected per token

    Returns:
        Output tensor, shape (B, S, D)
    """

    B, S, D = x.shape
    N = B * S

    # ---------------------------------------------------------
    # 1. Flatten tokens
    # ---------------------------------------------------------
    tokens = x.reshape(N, D)

    # ---------------------------------------------------------
    # 2. Gating
    # ---------------------------------------------------------
    logits = tokens @ Wg                         # (N, E)

    # Stable softmax
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)

    # ---------------------------------------------------------
    # 3. Top-k routing
    # ---------------------------------------------------------
    top_indices = np.argpartition(
        probs,
        -top_k,
        axis=-1
    )[:, -top_k:]                                # (N, K)

    top_probs = np.take_along_axis(
        probs,
        top_indices,
        axis=-1
    )                                             # (N, K)

    # Renormalize probabilities of selected experts
    top_probs /= top_probs.sum(axis=-1, keepdims=True)

    # ---------------------------------------------------------
    # 4. Flatten the routing decisions
    # ---------------------------------------------------------
    #
    # Example:
    #
    # top_indices:
    # [[2, 3],
    #  [0, 2],
    #  [1, 3]]
    #
    # becomes:
    #
    # expert_ids = [2, 3, 0, 2, 1, 3]
    #
    expert_ids = top_indices.ravel()             # (N*K,)

    token_ids = np.repeat(
        np.arange(N),
        top_k
    )                                             # (N*K,)

    weights = top_probs.ravel()                   # (N*K,)

    # ---------------------------------------------------------
    # 5. Group routing decisions by expert
    # ---------------------------------------------------------
    order = np.argsort(expert_ids)

    expert_ids = expert_ids[order]
    token_ids = token_ids[order]
    weights = weights[order]

    # ---------------------------------------------------------
    # 6. Compute only selected experts
    # ---------------------------------------------------------
    output = np.zeros((N, D), dtype=np.result_type(x, We, float))

    start = 0

    while start < len(expert_ids):

        expert = expert_ids[start]

        # Find the end of this expert's token group
        end = start + 1
        while end < len(expert_ids) and expert_ids[end] == expert:
            end += 1

        # Tokens routed to this expert
        ids = token_ids[start:end]

        # Their routing probabilities
        w = weights[start:end]

        # Compute expert only on these tokens
        expert_output = tokens[ids] @ We[expert]

        # Weighted accumulation
        output[ids] += w[:, None] * expert_output

        start = end

    return output.reshape(B, S, D)
