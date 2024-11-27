import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ScalingLayer(nn.Module):
    def __init__(self, scale_init_value=1, bias_init_value=0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([scale_init_value]))
        self.bias = nn.Parameter(torch.FloatTensor([bias_init_value]))

    def forward(self, input):
        return input * self.scale + self.bias

# FEEDFORWARD NEURAL NETWORK
class FNN(nn.Module):
    def __init__(self, dims, activation=nn.Sigmoid(), scaling=True):

        super(FNN, self).__init__()

        if scaling:
            if type(scaling) == bool:
                scaling = ScalingLayer()
        self.scaling = scaling

        self.dims = dims
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.act_fn = activation
    
    def forward(self, x):
        if self.scaling:
            x = self.scaling(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = self.act_fn(x)
        return x
    
