import torch


def to_numpy(tensor):
    # Detach tensor from gradient computation and move to CPU as numpy array
    if type(tensor) != torch.Tensor:
        return tensor
    return tensor.squeeze().detach().cpu().numpy()







