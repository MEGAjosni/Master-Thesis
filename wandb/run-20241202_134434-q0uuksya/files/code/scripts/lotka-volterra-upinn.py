# Torch
import torch
import torch.nn as nn
import torch.nn.init as init

# Custom imports
import os
import sys
sys.path.append('./')
from utils.NeuralNets import FNN
from utils.DataGenerators import LotkaVolterra
from utils.Utils import sample_with_noise

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

import wandb


# Generate data from Lotka-Volterra model
#   dx/dt = alpha*x - beta*x*y
#   dy/dt = gamma*x*y - delta*y

# # Parameters
# alpha, beta, gamma, delta = 2/3, 4/3, 1.0, 1.0
# x0, y0 = 1.0, 1.0
# alpha, beta, gamma, delta = 1.3, 0.9, 0.8, 1.8
# x0, y0 = 0.44249296, 4.6280594
alpha, beta, gamma, delta = 2/3, 4/3, 1.0, 1.0
x0, y0 = 1.0, 1.0
LV = LotkaVolterra(alpha, beta, gamma, delta, torch.tensor([x0, y0], dtype=torch.float32))

time_int = [0, 25]
N = 10000
t = torch.linspace(time_int[0], time_int[1], N)
X = LV.solve(t)
train_idx = torch.arange(0, 0.8*N, dtype=torch.long)
test_idx = torch.arange(0.8*N, N, dtype=torch.long)

# Sample subset and add noise
t_s, X_s = sample_with_noise(10, t[train_idx], X, epsilon=5e-3)

# Setup neural networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)
f_known = FNN(
    dims=[1, 16, 16, 16, 2],
    hidden_act=nn.Tanh(),
    output_act=nn.Softplus(),
    weight_init=init.xavier_normal_,
    bias_init=init.zeros_
).to(device)
f_unknown = FNN(
    dims=[2, 16, 16, 16, 2],
    hidden_act=nn.Tanh(),
    output_act=nn.Identity(),
    weight_init=init.xavier_normal_,
    bias_init=init.zeros_
).to(device)


# Setup Weights and Biases for online logging
wandb.init(
    project='Master-Thesis',
    name='UPINN-Lotka-Volterra',
    notes='Training f_known and f_unknown for 30000 epochs followed by training f_known for an additional 30000 with collacation points outside data range.',
    job_type='Train',
    save_code=True,
    config={
        "learning_rate": 1e-3,
        "Archtechture": "FNN",
        "Problem": "Lotka-Volterra",
        "Epochs": 60000,
        "Optimizer": "Adam",
    }
)
table = wandb.Table(columns=["Solution"])

optimizer = torch.optim.Adam([*f_known.parameters(), *f_unknown.parameters()], lr=wandb.config["learning_rate"])
# optimizer = torch.optim.LBFGS([*f_known.parameters(), *f_unknown.parameters()], lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)


# Weight scaling for the loss function
lambda1, lambda2, lambda3 = 1, 1, 1

# Move the data to the device and convert to float
X_f = X.to(device)
t_f = t.to(device).unsqueeze(-1).requires_grad_(True)
t_s = t_s.to(device).unsqueeze(-1)
X_s = X_s.to(device)
X0 = LV.X0.unsqueeze(0).to(device)

# Setup scaling layer
f_known.scale_fn = lambda t_: (t_-t.min())/(t.max()-t.min())
mu, sigma = 0, 2
epsilon = 1e-8
f_unknown.scale_fn = lambda x: (x-mu)/(sigma+epsilon)

# for epoch in range(wandb.config["Epochs"]):
for epoch in range(1):

    # Stop training of f_unknown after 30000 epochs and make collacation points outside the training data
    if epoch == 30000:
        optimizer = torch.optim.Adam(f_known.parameters(), lr=wandb.config["learning_rate"])
        t_f = torch.linspace(time_int[0], time_int[0]+2*(time_int[1]-time_int[0]), 2*N).to(device).unsqueeze(-1).requires_grad_(True)

    def closure():
        optimizer.zero_grad()
        
        # Initial condition loss
        t0 = torch.tensor([[0.0]], device=device).float()
        X0_pred = f_known(t0)
        print(t0)
        print(X0_pred)
        print(X0)
        ic_loss = nn.MSELoss()(X0_pred, X0)

        # Known dynamics loss
        X_f = f_known(t_f)
        x_f, y_f = X_f[:, 0:1], X_f[:, 1:2]
        dxdt = torch.autograd.grad(x_f, t_f, torch.ones_like(x_f), create_graph=True)[0]
        dydt = torch.autograd.grad(y_f, t_f, torch.ones_like(y_f), create_graph=True)[0]

        res_pred = f_unknown(X_f)
        res_x, res_y = res_pred[:, 0:1], res_pred[:, 1:2]
        dudt = torch.hstack([
            dxdt - LV.alpha * x_f + LV.beta * x_f * y_f - res_x,
            dydt + LV.delta * y_f - res_y
        ])
        pde_loss = torch.mean(dudt[:, 0] ** 2) + torch.mean(dudt[:, 1] ** 2)

        # Data loss
        X_pred = f_known(t_s)
        data_loss = nn.MSELoss()(X_pred, X_s)

        # Total loss
        loss = lambda1 * ic_loss + lambda2 * pde_loss + lambda3 * data_loss
        wandb.log({
            "Loss": loss.item(),
            "IC Loss": ic_loss.item(),
            "PDE Loss": pde_loss.item(),
            "Data Loss": data_loss.item()
        })
        print(f"Epoch {epoch}: Loss: {loss.item()}, IC Loss: {ic_loss.item()}, PDE Loss: {pde_loss.item()}, Data Loss: {data_loss.item()}")

        loss.backward(retain_graph=True)

        return loss

    optimizer.step(closure)  # Pass the closure to the optimizer

    # Plot solution
    if epoch % 1000 == 0:
        with torch.no_grad():

            # Evaluate the model
            X_pred = f_known(t.unsqueeze(-1).to(device))

            # Plot with plotly
            fig = go.Figure()
            fig.add_scatter(x=t, y=X[:, 0].cpu().numpy(), mode='lines', name='Prey', line=dict(dash='dash', color='green'))
            fig.add_scatter(x=t, y=X[:, 1].cpu().numpy(), mode='lines', name='Predator', line=dict(dash='dash', color='red'))
            fig.add_scatter(x=t, y=X_pred[:, 0].cpu().numpy(), mode='lines', name='Prey (pred)', line=dict(color='green'))
            fig.add_scatter(x=t, y=X_pred[:, 1].cpu().numpy(), mode='lines', name='Predator (pred)', line=dict(color='red'))
            # Add datapoints
            fig.add_scatter(x=t_s.squeeze().cpu().numpy(), y=X_s[:, 0].cpu().numpy(), mode='markers', name='Prey (data)', marker=dict(color='green', symbol='x'))
            fig.add_scatter(x=t_s.squeeze().cpu().numpy(), y=X_s[:, 1].cpu().numpy(), mode='markers', name='Predator (data)', marker=dict(color='red', symbol='x'))
            fig.update_layout(title=f"Lotka-Volterra Model (Epoch {epoch})")
            
            # Log figure to wandb
            wandb.log({"Solution": wandb.Plotly(fig)})

            # Plot missing terms
            res = f_unknown(X_pred).cpu()
            res_dx = res[:, 0]
            res_dy = res[:, 1]
            true_res_dx = torch.zeros_like(res_dx)
            true_res_dy = LV.gamma*X[:, 0]*X[:, 1]

            fig = go.Figure()
            fig.add_scatter(x=t, y=res_dx, mode='lines', name='Residual Prey', line=dict(color='green'))
            fig.add_scatter(x=t, y=res_dy, mode='lines', name='Residual Predator', line=dict(color='red'))
            fig.add_scatter(x=t, y=true_res_dx, mode='lines', name='Prey: 0', line=dict(dash='dash', color='green'))
            fig.add_scatter(x=t, y=true_res_dy, mode='lines', name='Predator: γ*x*y', line=dict(dash='dash', color='red'))
            fig.update_layout(title=f"Lotka-Volterra Missing Terms (Epoch {epoch})")

            # Log figure to wandb
            wandb.log({"Missing Terms": wandb.Plotly(fig)})


# Save the model
torch.save(f_known.state_dict(), 'models/lotka-volterra/f_known1.pt')
torch.save(f_unknown.state_dict(), 'models/lotka-volterra/f_unknown1.pt')