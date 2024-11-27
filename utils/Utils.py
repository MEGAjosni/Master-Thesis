import torch


def to_numpy(tensor):
    # Detach tensor from gradient computation and move to CPU as numpy array
    if type(tensor) != torch.Tensor:
        return tensor
    return tensor.squeeze().detach().cpu().numpy()


def sample_with_noise(N, t, X, epsilon=5e-3):

    # Check if the shapes are correct and feasible amount of points are requested
    assert len(X) != len(t) or N <= len(t), "Invalid shapes or N"

    # Calculate the mean of the data
    X_bar = torch.mean(X, dim=0)

    # Sample N evenly spaced points from the data
    idx = torch.linspace(0, len(t)-1, N, dtype=torch.int)
    t, X = t[idx], X[idx]

    # Add noise to the data
    X_noise = X + epsilon * X_bar * torch.randn(*X.shape)

    return t, X_noise