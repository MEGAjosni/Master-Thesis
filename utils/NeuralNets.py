import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# FEEDFORWARD NEURAL NETWORK
class FNN(nn.Module):
    def __init__(self, dims, hidden_act=nn.Tanh(), output_act=nn.Identity(), weight_init=None, bias_init=None, scale_fn=lambda x: x):

        super(FNN, self).__init__()

        self.dims = dims
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.hidden_act_fn = hidden_act
        self.output_act_fn = output_act
        self.scale_fn = scale_fn
        self.inititialize_weights(weight_init, bias_init)


    def inititialize_weights(self, weight_init, bias_init):
        if weight_init:
            for layer in self.layers:
                weight_init(layer.weight)
        if bias_init:
            for layer in self.layers:
                bias_init(layer.bias)

    
    def forward(self, x):
        x = self.scale_fn(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = self.hidden_act_fn(x)
        return self.output_act_fn(x)
    

# class ResNet(nn.Module):
#     def __init__(self, dims, hidden_act=nn.Tanh(), output_act=nn.Identity(), weight_init=None, bias_init=None, scale_fn=lambda x: x):
#         super(ResNet, self).__init__()

#         self.dims = dims
#         self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
#         self.hidden_act_fn = hidden_act
#         self.output_act_fn = output_act
#         self.scale_fn = scale_fn
#         self.inititialize_weights(weight_init, bias_init)


#     def inititialize_weights(self, weight_init, bias_init):
#         if weight_init:
#             for layer in self.layers:
#                 weight_init(layer.weight)
#         if bias_init:
#             for layer in self.layers:
#                 bias_init(layer.bias)

    
#     def forward(self, x):
#         x = self.scale_fn(x)
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < len(self.layers)-1:
#                 x = self.hidden_act_fn(x)
#         return self.output_act_fn(x) + x
    
