# Torch
import torch

# Custom imports
import sys
sys.path.append('./')
from utils.architectures import FNN, KAN
from utils.DataGenerators import LotkaVolterra
from utils.Utils import sample_with_noise
from utils.train import train

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

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
    z_b=torch.tensor([[0.0]]),
    U_b=LV.X0.unsqueeze(0),
    z_d=t_d.unsqueeze(-1),
    U_d=X_d,
    z_c=t[train_idx].unsqueeze(-1).requires_grad_(True),
)

# Define model architectures with sweep params
hidden = [16] * 4
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

# Define pde residual
def pde_residual(X, res, t, params):
    dUdt = torch.cat([
        torch.autograd.grad(outputs=X[:, i], inputs=t, grad_outputs=torch.ones_like(X[:, i]), create_graph=True)[0]
        for i in range(X.shape[1])
    ], dim=-1)
    return torch.stack([
        dUdt[:, 0] - params[0]*X[:, 0] + params[1]*X[:, 0]*X[:, 1] - res[:, 0],
        dUdt[:, 1] + params[2]*X[:, 1] - res[:, 1]
    ], dim=-1)

# Make plotting function

def plot_solution(u, G):
    device = next(u.parameters()).device # Get the device of the model
    plots = dict()

    # Evaluate the model
    X_pred = u(t.unsqueeze(-1).to(device))

    # Plot with plotly
    fig = go.Figure()
    fig.add_scatter(x=t, y=X[:, 0].cpu().numpy(), mode='lines', name='Prey', line=dict(dash='dash', color='green'))
    fig.add_scatter(x=t, y=X[:, 1].cpu().numpy(), mode='lines', name='Predator', line=dict(dash='dash', color='red'))
    fig.add_scatter(x=t, y=X_pred[:, 0].cpu().numpy(), mode='lines', name='Prey (pred)', line=dict(color='green'))
    fig.add_scatter(x=t, y=X_pred[:, 1].cpu().numpy(), mode='lines', name='Predator (pred)', line=dict(color='red'))
    # Add datapoints
    fig.add_scatter(x=t_d.squeeze().cpu().numpy(), y=X_d[:, 0].cpu().numpy(), mode='markers', name='Prey (data)', marker=dict(color='green', symbol='x'))
    fig.add_scatter(x=t_d.squeeze().cpu().numpy(), y=X_d[:, 1].cpu().numpy(), mode='markers', name='Predator (data)', marker=dict(color='red', symbol='x'))
    fig.update_layout(title=f"Lotka-Volterra Model")
    
    # Log figure to wandb
    plots["Solution"] = fig

    # Plot missing terms
    res = G(X_pred).cpu()
    res_dx = res[:, 0]
    res_dy = res[:, 1]
    true_res_dx = torch.zeros_like(res_dx)
    true_res_dy = LV.gamma*X[:, 0]*X[:, 1]

    fig = go.Figure()
    fig.add_scatter(x=t, y=res_dx, mode='lines', name='Residual Prey', line=dict(color='green'))
    fig.add_scatter(x=t, y=res_dy, mode='lines', name='Residual Predator', line=dict(color='red'))
    fig.add_scatter(x=t, y=true_res_dx, mode='lines', name='Prey: 0', line=dict(dash='dash', color='green'))
    fig.add_scatter(x=t, y=true_res_dy, mode='lines', name='Predator: γ*x*y', line=dict(dash='dash', color='red'))
    fig.update_layout(title=f"Lotka-Volterra Missing Terms")

    # Log figure to wandb
    plots["Missing Terms"] = fig

    return plots


# Train the model
train(
    u, G, data,
    torch.tensor([alpha, beta, delta], dtype=torch.float32),
    pde_residual,
    optimizer=torch.optim.AdamW,
    optimizer_args=dict(lr=1e-3, weight_decay=1e-4),
    epochs=1000,
    loss_tol_stop=1e-5,
    log_wandb=dict(log=True, name='LV-UPINN-FNN', project='Master-Thesis', log_plots=True, plot_interval=1000, plot_fn=plot_solution)
)