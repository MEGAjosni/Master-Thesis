# Torch
import torch

# Custom imports
import os
import sys
sys.path.append('./')
from utils.NeuralNets import FNN, KAN
from utils.DataGenerators import LotkaVolterra
from utils.Utils import sample_with_noise, SoftAdapt

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

import wandb

exp_name = "lotka-volterra-upinn-FNN"



# # Parameters
# alpha, beta, gamma, delta = 2/3, 4/3, 1.0, 1.0
# x0, y0 = 1.0, 1.0
# alpha, beta, gamma, delta = 1.3, 0.9, 0.8, 1.8
# x0, y0 = 0.44249296, 4.6280594

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

###############################################
### Setup the neural networks for training  ###
###############################################
u = FNN(
    dims=[1, 16, 16, 16, 2],
    hidden_act=torch.nn.Tanh(),
    output_act=torch.nn.Softplus(),
)
G = FNN(
    dims=[2, 16, 16, 16, 2],
    hidden_act=torch.nn.Tanh(),
    output_act=torch.nn.ReLU(),
)

# Setup scaling layer
u.scale_fn = lambda t_: (t_-t.min())/(t.max()-t.min())
mu, sigma = 0, 2
epsilon = 1e-8
G.scale_fn = lambda x: (x-mu)/(sigma+epsilon)


##############################################
### Setup WandB for logging and monitoring ###
##############################################
wandb.init(
    project='Master-Thesis',
    group='UPINN',
    name=exp_name,
    notes='Training f_known and f_unknown for 30000 epochs followed by training f_known for an additional 30000 with collacation points outside data range.',
    job_type='Train',
    save_code=True,
    config={
        "learning_rate": 1e-3,
        "Archtechture": "FNN",
        "Problem": "Lotka-Volterra",
        "Epochs": 30000,
        "Optimizer": "AdamW",
    }
)
table = wandb.Table(columns=["Solution"])

#####################
### Training loop ###
#####################
def train(
        u: torch.nn.Module,
        G: torch.nn.Module,
        data: dict,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 1000,
        loss_tol_stop: float = None,    # Stop training if loss is below this value
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        beta_softadapt: float = 0.1,
        plotting: dict = dict(log_plot=False, plot_interval=1000),
        save_model: dict = dict(save=False, path='models/lotka-volterra', name='LV-UPINN'),
):

    print("Beginning training...")
    print("Running on:", device)

    # Move the model to the device
    u.to(device); G.to(device)

    # Unpack data and move to device        
    t_b = data["t_b"].to(device)                      # Boundary points
    X_b = data["X_b"].to(device)

    t_d = data["t_d"].to(device)                      # Data points
    X_d = data["X_d"].to(device)

    t_c = data["t_c"].to(device).requires_grad_(True) # Collocation points

    # Initialize parameters
    theta = torch.nn.Parameter(torch.zeros(3))

    # Initialize optimizer
    optimizer = optimizer([*u.parameters(), *G.parameters(), theta], lr=lr, weight_decay=weight_decay)

    # Initialize previous losses for SoftAdapt
    prev_losses = torch.zeros(3).to(device)

    for epoch in range(epochs):

        # Stop training of f_unknown after 30000 epochs and make collacation points outside the training data
        # if epoch == 30000:
        #     optimizer = torch.optim.Adam(f_known.parameters(), lr=lr, weight_decay=weight_decay)
        #     t_f = t.to(device).unsqueeze(-1).requires_grad_(True)

        def closure():
            optimizer.zero_grad()

            # Boundary condition loss
            bc_loss = torch.nn.MSELoss()(u(t_b), X_b)

            # Data loss
            data_loss = torch.nn.MSELoss()(u(t_d), X_d)

            # Known PDE loss
            X_c = u(t_c)
            x_c, y_c = X_c.T.unsqueeze(-1)
            dxdt = torch.autograd.grad(x_c, t_c, torch.ones_like(x_c), create_graph=True)[0]
            dydt = torch.autograd.grad(y_c, t_c, torch.ones_like(y_c), create_graph=True)[0]

            res_x, res_y = G(X_c).T.unsqueeze(-1)
            dudt = torch.hstack([
                dxdt - theta[0] * x_c + theta[1] * x_c * y_c - res_x,
                dydt + theta[2] * y_c - res_y
            ])
            pde_loss = torch.nn.MSELoss()(dudt, torch.zeros_like(dudt))


            # Total loss
            cur_losses = torch.stack([bc_loss, pde_loss, data_loss])
            lambda_ = SoftAdapt(cur_losses, prev_losses, beta=beta_softadapt)   # SoftAdapt weights
            loss = torch.dot(lambda_, cur_losses)

            # Log losses to wandb
            wandb.log({
                "Loss": loss.item(),
                "BC Loss": bc_loss.item(),
                "PDE Loss": pde_loss.item(),
                "Data Loss": data_loss.item()
            })

            # Backpropagate
            loss.backward(retain_graph=True)

            return loss, cur_losses.clone()

        loss, prev_losses = optimizer.step(closure)  # Pass the closure to the optimizer


        # Plot solution
        if plotting["log_plots"] and epoch % plotting["plot_interval"] == 0:
            with torch.no_grad():

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
                fig.update_layout(title=f"Lotka-Volterra Model (Epoch {epoch})")
                
                # Log figure to wandb
                wandb.log({"Solution": wandb.Plotly(fig)})

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
                fig.update_layout(title=f"Lotka-Volterra Missing Terms (Epoch {epoch})")

                # Log figure to wandb
                wandb.log({"Missing Terms": wandb.Plotly(fig)})
        

        if loss_tol_stop:
            if loss < loss_tol_stop:
                print("Loss below tolerance. Stopping training.")
                break
    
    print("Training complete.")

    # Save the model
    if save_model["save"]:
        print("Saving model...")
        torch.save(u.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_u.pth'))
        torch.save(G.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_G.pth'))

train(u, G, data, optimizer=torch.optim.AdamW, lr=wandb.config["learning_rate"], epochs=wandb.config["Epochs"], plotting=dict(log_plots=True, plot_interval=1000), save_model=dict(save=False, path='models/lotka-volterra', name='LV-UPINN'))