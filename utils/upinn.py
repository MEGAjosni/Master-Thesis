# Torch
import torch

# Custom imports
import os
import sys
sys.path.append('./')
from utils.Utils import SoftAdapt

# Weights and Biases
import wandb
os.environ["WANDB_MODE"] = "online" # Only log to WandB online
import  tempfile
folder_temp = tempfile.TemporaryDirectory()
os.chmod(folder_temp.name, 0o777)


class UPINN:

    def __init__(self, u, G):
        self.u = u
        self.G = G        




    def train(
        u: torch.nn.Module,
        G: torch.nn.Module,
        data: dict,
        params: torch.Tensor,
        pde_residual, # Function that returns the PDE residual
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_args: dict = dict(lr=1e-3, weight_decay=0.0),
        scheduler: torch.optim.Optimizer = None,
        scheduler_args: dict = None,
        epochs: int = 1000,
        loss_tol_stop: float = None,    # Stop training if loss is below this value
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        beta_softadapt: float = 0.1,
        save_model: dict = dict(save=False, path='models/lotka-volterra', name='LV-UPINN'),
        log_wandb: dict = dict(log=False, name='UPINN', project='Master-Thesis', log_plots=False, plot_interval=1000, plot_fn=None)
):

    print("Beginning training...")
    print("Running on:", device)

    # Move the model to the device
    u.to(device); G.to(device)

    # Unpack data and move to device        
    z_b = data["z_b"].to(device)                      # Boundary points
    U_b = data["U_b"].to(device)

    z_d = data["z_d"].to(device)                      # Data points
    U_d = data["U_d"].to(device)

    z_c = data["z_c"].to(device).requires_grad_(True) # Collocation points

    
    # Initialize optimizer
    optimizer = optimizer([*u.parameters(), *G.parameters()], **optimizer_args)
    
    # Setup learning rate scheduler
    if scheduler is not None:
        scheduler = scheduler(**scheduler_args)

    # Initialize parameters
    params = params.to(device)
    if type(params) == torch.nn.Parameter:
        params.requires_grad_(True)
        optimizer.add_param_group({'params': params})

    # Setup WandB logging
    if log_wandb["log"]:
        wandb.init(project=log_wandb["project"], name=log_wandb["name"], dir=folder_temp)
        wandb.watch(u)
        wandb.watch(G)

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
            bc_loss = torch.nn.MSELoss()(u(z_b), U_b)

            # Data loss
            data_loss = torch.nn.MSELoss()(u(z_d), U_d) if U_d.numel() > 0 else torch.tensor(0).to(device) # No data

            # PDE loss
            U_c = u(z_c)
            res = G(U_c)
            pde_loss = torch.nn.MSELoss()(pde_residual(U_c, res, z_c, params), torch.zeros_like(U_c))

            # Total loss
            cur_losses = torch.stack([bc_loss, pde_loss, data_loss])
            lambda_ = SoftAdapt(cur_losses, prev_losses, beta=beta_softadapt)   # SoftAdapt weights
            loss = torch.dot(lambda_, cur_losses)

            # Log losses to wandb
            if log_wandb["log"]:
                wandb.log({
                    "Epoch": epoch,
                    "Loss": loss.item(),
                    "BC Loss": bc_loss.item(),
                    "PDE Loss": pde_loss.item(),
                    "Data Loss": data_loss.item(),
                })

            # Backpropagate
            loss.backward(retain_graph=True)

            return loss, cur_losses.clone()

        loss, prev_losses = optimizer.step(closure)  # Pass the closure to the optimizer
        scheduler.step() if scheduler is not None else None


        # Plot solution
        if log_wandb['log']:
            if log_wandb["log_plots"] and epoch % log_wandb["plot_interval"] == 0:
                with torch.no_grad():
                    plots = log_wandb["plot_fn"](u, G)
                    for name, fig in plots.items():
                        wandb.log({name: wandb.Plotly(fig)})
        

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