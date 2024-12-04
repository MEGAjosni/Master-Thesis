# Torch
import torch

# Custom imports
import os
import sys
sys.path.append('./')
from utils.NeuralNets import FNN, KAN
from utils.DataGenerators import LotkaVolterra
from utils.Utils import sample_with_noise, SoftAdapt
from scripts.TrainLV import train

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

import wandb
exp_name = "lotka-volterra-upinn-FNN"



# Sweep configuration
sweep_config = {
    'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'Epoch',  # Metric to optimize (logged in WandB)
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-4, 1e-3, 1e-2]  # Test these values
        },
        'hidden_layers': {
            'values': [1, 2, 3, 4]  # Number of hidden layers
        },
        'hidden_size': {
            'values': [8, 16, 32, 64]  # Size of each hidden layer
        },
        'weight_decay': {
            'values': [0.0, 1e-5, 1e-4]  # Additional optional parameter
        }
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project='Master-Thesis')  # Replace with your WandB username or org


###############################################
### Generate data from Lotka-Volterra model ###
###############################################
###   dx/dt = alpha*x - beta*x*y            ###
###   dy/dt = gamma*x*y - delta*y           ###
###############################################
alpha, beta, gamma, delta = 2/3, 4/3, 1.0, 1.0
x0, y0 = 1.0, 1.0
LV = LotkaVolterra(alpha, beta, gamma, delta, torch.tensor([x0, y0], dtype=torch.float32))

time_int = [0, 25]
train_test = 0.8
N = 800
t = torch.linspace(time_int[0], time_int[1], N)
X = LV.solve(t)
train_idx = torch.arange(0, train_test*N, dtype=torch.long)
test_idx = torch.arange(train_test*N, N, dtype=torch.long)

# Sample subset and add noise
t_d, X_d = sample_with_noise(10, t[train_idx], X, epsilon=5e-3)

# Move the data to the device and convert to float

data = dict(
    t_b=torch.tensor([[0.0]]),
    X_b=LV.X0.unsqueeze(0),
    t_d=t_d.unsqueeze(-1),
    X_d=X_d,
    t_c=t[train_idx].unsqueeze(-1).requires_grad_(True),
)


# Sweep train function
def sweep_train():
    wandb.init()  # Initialize the run
    config = wandb.config  # Access sweep parameters

    # Define model architectures with sweep params
    hidden = [config.hidden_size] * config.hidden_layers
    u = FNN(
        dims=[1, *hidden, 2],
        hidden_act=torch.nn.Tanh(),
        output_act=torch.nn.Softplus(),
    )
    G = FNN(
        dims=[2, *hidden, 2],
        hidden_act=torch.nn.Tanh(),
        output_act=torch.nn.ReLU(),
    )

    # Setup scaling layer
    u.scale_fn = lambda t_: (t_-t.min())/(t.max()-t.min())
    mu, sigma = 0, 2
    epsilon = 1e-8
    G.scale_fn = lambda x: (x-mu)/(sigma+epsilon)

    # Train the model
    train(
        u, G, data,
        optimizer=torch.optim.AdamW,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=30000,
        plotting=dict(log_plots=False, plot_interval=1000),
        loss_tol_stop=1e-5,
    )

# Run the sweep
wandb.agent(sweep_id, function=sweep_train, count=10)  # Adjust count as needed
