# Torch
import torch

# Custom imports
import sys
sys.path.append('./')
from utils.NeuralNets import FNN, KAN
from utils.DataGenerators import LotkaVolterra
from utils.Utils import sample_with_noise
from scripts.train import train

# Plotly
import plotly.graph_objects as go
import plotly.express as px

###############################################
### Generate data from Lotka-Volterra model ###
###############################################
###   dx/dt = alpha*x - beta*x*y            ###
###   dy/dt = gamma*x*y - delta*y           ###
###############################################

nu = 1/1000*torch.pi
N_bc = 100
N_c = 10000

x_ic = torch.linspace(-1, 1, N_bc).unsqueeze(-1)
t_ic = torch.zeros_like(x_ic)
z_ic = torch.cat([t_ic, x_ic], dim=-1)
U_ic = -torch.sin(torch.pi * z_ic[:, 1:2])

t_bc = torch.linspace(0, 1, N_bc).unsqueeze(-1)
x_bc = torch.cat([-torch.ones(N_bc//2), torch.ones(N_bc//2)], dim=0).unsqueeze(-1)
z_bc = torch.cat([t_bc, x_bc], dim=-1)
U_bc = torch.zeros_like(x_bc)

sobel_engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
x_c = (sobel_engine.draw(N_c, dtype=torch.float32)*2 - 1).requires_grad_(True)
t_c = (sobel_engine.draw(N_c, dtype=torch.float32)).requires_grad_(True)
z_c = torch.cat([t_c, x_c], dim=-1)

# No Data
z_d = torch.empty(0, 2)
U_d = torch.empty(0, 1)

data = dict(
    z_b=torch.vstack([z_ic, z_bc]),
    U_b=torch.vstack([U_ic, U_bc]),
    z_d=z_d,
    U_d=U_d,
    z_c=z_c.requires_grad_(True),
)

# Define model architectures
hidden = [20] * 3
u = FNN(
    dims=[2, *hidden, 1],
    hidden_act=torch.nn.Tanh(),
)
G = FNN(
    dims=[1, *hidden, 1],
    hidden_act=torch.nn.Tanh(),
    output_act=torch.nn.ReLU(),
)

# Setup scaling layer
# u.scale_fn = lambda t_: (t_-t.min())/(t.max()-t.min())
# mu, sigma = 0, 2
# epsilon = 1e-8
# G.scale_fn = lambda x: (x-mu)/(sigma+epsilon)

# Define pde residual
def pde_residual(U, res, z, params):
    Ut, Ux = torch.autograd.grad(outputs=U, inputs=z, grad_outputs=torch.ones_like(U), create_graph=True, retain_graph=True)[0].T
    Uxx = torch.autograd.grad(outputs=Ux, inputs=z, grad_outputs=torch.ones_like(Ux), create_graph=True)[0].T[1]
    return (Ut + U.squeeze() * Ux - params[0] * Uxx).unsqueeze(-1)

# Make plotting function #TODO: Make this work Properly
def plot_solution(u, G):
    device = next(u.parameters()).device # Get the device of the model
    plots = dict()

    # Evaluate the model on a grid
    t = torch.linspace(0, 1, 1600)
    x = torch.linspace(-1, 1, 400)
    T, X = torch.meshgrid(t, x, indexing='ij')
    T, X = T.reshape(-1, 1), X.reshape(-1, 1)
    Z = torch.cat((T, X), dim=1)

    U_pred = u(Z.to(device)).detach().cpu().numpy()

    # Plot the results as a heatmap using Plotly
    fig = px.imshow(
        U_pred.reshape(1600, 400).T,
        x=t,
        y=x,
        labels=dict(x="t", y="x", color="u"),
        title="Burgers' Equation Solution",
        color_continuous_scale='Viridis'
    )

    plots["Solution"] = fig

    return plots


# Train the model
train(
    u, G, data,
    torch.tensor([nu], dtype=torch.float32),
    pde_residual,
    optimizer=torch.optim.Adam,
    optimizer_args=dict(),
    epochs=5000,
    loss_tol_stop=1e-5,
    log_wandb=dict(log=True, name='BURGERS-UPINN-FNN', project='Master-Thesis', log_plots=True, plot_interval=1000, plot_fn=plot_solution)
)