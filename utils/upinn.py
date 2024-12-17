# Torch
import torch

# Custom imports
import os
from tqdm import tqdm
from .BVP import BVP

# Weights and Biases
import wandb
os.environ["WANDB_MODE"] = "online" # Only log to WandB online
import  tempfile
folder_temp = tempfile.TemporaryDirectory()
os.chmod(folder_temp.name, 0o777)



def SoftAdapt(cur_losses, prev_losses, beta=0, loss_weigthed=False):
    f = cur_losses.detach()
    fm1 = prev_losses.detach()
    s = f - fm1
    if loss_weigthed:
        return f*torch.exp(beta*s) / torch.sum(f*torch.exp(beta*s))
    else:
        return torch.nn.functional.softmax(beta*s, dim=0)



class DeadZoneLinear(torch.nn.Module):
    def __init__(self, a=0.1):
        super(DeadZoneLinear, self).__init__()
        self.a = a  # The range [-a, a] where the function outputs 0

    def forward(self, x):
        return torch.where(x > self.a, x - self.a, 
                           torch.where(x < -self.a, x + self.a, torch.tensor(0.0, device=x.device)))




class UPINN:

    def __init__(self,
        u: torch.nn.Module,
        G: torch.nn.Module,
        bvp: BVP, # Parameters to be estimated are also included in the BVP object
    ):
        """
        u: torch.nn.Module
            Neural Network estimating the solution of the PDE

        G: torch.nn.Module
            Neural Network estimating the PDE residual

        bvp: BVP
            Boundary Value Problem object: Can evaluate the PDE residual and boundary residual
            Has an attribute for parameters (params: dict)
                Parameters of the BVP.
                If some parameters are to be estimated, they should be torch.nn.Parameter. As an example:
                params = {
                    alpha: torch.nn.Parameter(torch.tensor(1.0)),
                    beta: 2.3,
                    }
                This will estimate alpha from an initial guess of 1.0, while beta will be kept constant at 2.3.
        """
    
        self.u = u
        self.G = G
        self.bvp = bvp


    def train(
            self,
            data_points: torch.Tensor = None,
            data_target: torch.Tensor = None,
            boundary_points: torch.Tensor = None,
            collocation_points: torch.Tensor = None,
            lambda_reg: float = 0.0,
            priotize_pde: float = 1.0,
            boundary_refiner: callable = lambda z, loss: z,
            collocation_refiner: callable = lambda z, loss: z,
            loss_fn_bc: torch.nn.Module = torch.nn.MSELoss(),
            loss_fn_data: torch.nn.Module = torch.nn.MSELoss(),
            loss_fn_pde: torch.nn.Module = torch.nn.MSELoss(),
            optimizer: torch.optim.Optimizer = torch.optim.AdamW,
            optimizer_args: dict = dict(lr=1e-3, weight_decay=0.0),
            scheduler: torch.optim.Optimizer = None,
            scheduler_args: dict = None,
            epochs: int = 1000,
            loss_tol: float = None,    # Stop training if loss is below this value
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            beta_softadapt: float = 0.0,
            save_model: dict = dict(save=False, path='models/lotka-volterra', name='LV-UPINN'),
            log_wandb: dict = dict(name='UPINN', project='Master-Thesis', plotter=None, plot_interval=1000)
    ):

        print("Beginning training...")
        print("Running on:", device)

        u = self.u; G = self.G; bvp = self.bvp 

        # Move the model and data to the device
        u.to(device); G.to(device); bvp.to(device)
        data_points = data_points.to(device) if data_points is not None else None
        data_target = data_target.to(device) if data_target is not None else None
        boundary_points = boundary_points.to(device) if boundary_points is not None else None
        collocation_points = collocation_points.to(device) if collocation_points is not None else None
        
        # Initialize optimizer
        params = [param for param in bvp.params.values() if isinstance(param, torch.nn.Parameter)]
        optimizer = optimizer([*u.parameters(), *G.parameters(), *params], **optimizer_args)
        
        # Setup learning rate scheduler
        if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_args)

        # Setup WandB logging
        if log_wandb is not None:
            wandb.init(project=log_wandb["project"], name=log_wandb["name"])
            wandb.watch(u)
            wandb.watch(G)
            epoch_iterator = range(epochs)
        else:
            epoch_iterator = tqdm(range(epochs))

        # Initialize previous losses for SoftAdapt
        prev_losses = torch.zeros(3).to(device)
        loss = 1.0

        for epoch in epoch_iterator:

            def closure():
                optimizer.zero_grad()

                # Boundary condition loss
                bc_residual = bvp.g(boundary_points, u(boundary_points)) if boundary_points is not None else torch.tensor(0).to(device) # No boundary conditions
                bc_loss = loss_fn_bc(bc_residual, torch.zeros_like(bc_residual)) # Enforce boundary conditions

                # Data loss
                data_loss = loss_fn_data(u(data_points), data_target) if data_points is not None else torch.tensor(0).to(device) # No data
                # data_loss = torch.mean(torch.mean(DeadZoneLinear(0.05)(torch.abs(u(data_points) - data_target)), dim=0))

                # PDE loss
                U_c = u(collocation_points)
                res = G(U_c)
                pde_loss = priotize_pde*loss_fn_pde(bvp.f(collocation_points, u(collocation_points)), res)

                # Regularization loss
                reg_loss = torch.mean(torch.abs(res))

                # Total loss
                cur_losses = torch.stack([bc_loss, pde_loss, data_loss])
                lambda_ = SoftAdapt(cur_losses, prev_losses, beta=beta_softadapt)   # SoftAdapt weights
                loss = torch.dot(lambda_, cur_losses)

                if log_wandb is not None:
                    wandb.log({
                        "Epoch": epoch,
                        "LR": optimizer.param_groups[0]['lr'],
                        "Loss": loss.item(),
                        "BC Loss": bc_loss.item(),
                        "PDE Loss": pde_loss.item(),
                        "Data Loss": data_loss.item(),
                        "Reg Loss": reg_loss.item(),
                        **{key: value.item() for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
                        "lambda_bc": lambda_[0].item(),
                        "lambda_pde": lambda_[1].item(),
                        "lambda_data": lambda_[2].item(),
                        "lambda_reg": lambda_reg,
                    })
                else:
                    epoch_iterator.set_postfix({
                        "Loss": round(loss.item(), 6),
                        "BC Loss": round(bc_loss.item(), 6),
                        "PDE Loss": round(pde_loss.item(), 6),
                        "Data Loss": round(data_loss.item(), 6),
                        "Reg Loss": round(reg_loss.item(), 6),
                        "LR_reg": round(lambda_reg, 3),
                        **{key: round(value.item(), 3) for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
                    })

                # Backpropagate
                nn_params = [param for param in bvp.params.values() if isinstance(param, torch.nn.Parameter)]
                lambda_param = 0.01
                params_loss = torch.sum(torch.nn.ReLU()(-torch.stack(nn_params))) if len(nn_params) > 0 else torch.tensor(0).to(device)
                (loss + lambda_param*params_loss + lambda_reg*reg_loss).backward(retain_graph=True)

                return loss, bc_loss, pde_loss, data_loss, cur_losses.clone()

            # Perform optimization step and update learning rate
            loss, bc_loss, pde_loss, data_loss, prev_losses = optimizer.step(closure)  # Pass the closure to the optimizer
            scheduler.step(loss.item()) if scheduler is not None else None

            # Refine boundary and collocation points
            boundary_points = boundary_refiner(boundary_points, bc_loss)
            collocation_points = collocation_refiner(collocation_points, pde_loss)

            # Plot solution
            if log_wandb is not None:
                if log_wandb["plotter"] is not None and epoch % log_wandb["plot_interval"] == 0:
                    with torch.no_grad():
                        plots = log_wandb["plotter"](u, G)
                        for name, fig in plots.items():
                            wandb.log({name: wandb.Plotly(fig)})
            

            if loss_tol is not None:
                if loss.item() < loss_tol:
                    print(f"Loss below tolerance at epoch {epoch}. Terminating training.")
                    break
        
        print("Training complete.")

        # Save the model
        if save_model["save"]:
            print("Saving model...")
            torch.save(u.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_u.pth'))
            torch.save(G.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_G.pth'))








# class UPINN:

#     def __init__(self,
#         u: torch.nn.Module,
#         G: torch.nn.Module,
#         bvp: BVP, # Parameters to be estimated are also included in the BVP object
#     ):
#         """
#         u: torch.nn.Module
#             Neural Network estimating the solution of the PDE

#         G: torch.nn.Module
#             Neural Network estimating the PDE residual

#         bvp: BVP
#             Boundary Value Problem object: Can evaluate the PDE residual and boundary residual
#             Has an attribute for parameters (params: dict)
#                 Parameters of the BVP.
#                 If some parameters are to be estimated, they should be torch.nn.Parameter. As an example:
#                 params = {
#                     alpha: torch.nn.Parameter(torch.tensor(1.0)),
#                     beta: 2.3,
#                     }
#                 This will estimate alpha from an initial guess of 1.0, while beta will be kept constant at 2.3.
#         """
    
#         self.u = u
#         self.G = G
#         self.bvp = bvp


#     def train(self,
#         data_points: torch.Tensor = None,
#         data_target: torch.Tensor = None,
#         boundary_points: torch.Tensor = None,
#         collocation_points: torch.Tensor = None,
#         boundary_refiner: callable = lambda z, loss: None,
#         collocation_refiner: callable = lambda z, loss: None,
#         loss_fn: torch.nn.Module = torch.nn.MSELoss(),
#         optimizer: torch.optim.Optimizer = torch.optim.Adam,
#         optimizer_args: dict = dict(lr=1e-3, weight_decay=0.0),
#         scheduler: torch.optim.Optimizer = None,
#         scheduler_args: dict = dict(),
#         scheduler_extend: torch.optim.Optimizer = None,
#         scheduler_extend_args: dict = dict(),
#         epochs: int = 30000,
#         loss_tol: float = None,    # Stop training if loss is below this value
#         device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         beta_softadapt: float = 0.1,
#         save_model: dict = dict(save=False, path='models/lotka-volterra', name='LV-UPINN'),
#         log_wandb: dict = dict(name='UPINN', project='Master-Thesis', plotter=None, plot_interval=1000)
#     ):
#         u = self.u; G = self.G; bvp = self.bvp

#         print("Beginning training...")
#         print("Running on:", device)

#         # Move the model and data to the device
#         u.to(device); G.to(device); bvp.to(device)
#         boundary_points = boundary_points.to(device) if boundary_points is not None else None
#         collocation_points = collocation_points.to(device) if collocation_points is not None else None
#         data_points = data_points.to(device) if data_points is not None else None
#         data_target = data_target.to(device) if data_target is not None else None

#         # Initialize optimizer
#         optimizer_ = optimizer([*u.parameters(), *G.parameters(), *[param for param in bvp.params.values() if isinstance(param, torch.nn.Parameter)]], **optimizer_args)
        
#         # Setup learning rate scheduler
#         scheduler_ = scheduler(optimizer_, **scheduler_args) if scheduler is not None else None

#         # Setup WandB logging
#         if log_wandb is not None:
#             wandb.init(project=log_wandb["project"], name=log_wandb["name"], dir=folder_temp)
#             wandb.watch(u)
#             wandb.watch(G)

#         # Initialize previous losses for SoftAdapt
#         prev_losses = torch.zeros(3).to(device)

#         for epoch in range(epochs):

#             def closure(
#                     # Make sure the points are in the scope of the closure
#                     boundary_points=boundary_points,
#                     collocation_points=collocation_points,
#                     data_points=data_points,
#                     data_target=data_target
#                 ):

#                 optimizer_.zero_grad()

#                 # Boundary condition loss
#                 if boundary_points is not None:
#                     bc_residual = bvp.g(boundary_points, u(boundary_points))
#                     bc_loss = loss_fn(bc_residual, torch.zeros_like(bc_residual)) # Enforce boundary conditions
#                 else:
#                     bc_loss = torch.tensor(0).to(device) # No boundary conditions
#                 # bc_loss = loss_fn(u(system.z_b), system.U_b) if system.U_b.numel() > 0 else torch.tensor(0).to(device) # No boundary conditions

#                 # Data loss
#                 data_loss = loss_fn(u(data_points), data_target) if data_points is not None else torch.tensor(0).to(device) # No data
#                 # data_loss = loss_fn(u(system.z_d), system.U_d) if system.U_d.numel() > 0 else torch.tensor(0).to(device) # No data

#                 # PDE loss
#                 if collocation_points is not None:
#                     pde_residual = bvp.f(collocation_points, u(collocation_points))
#                     pde_loss = loss_fn(pde_residual, G(u(collocation_points))) # Enforce PDE residual
#                 else:
#                     pde_loss = torch.tensor(0).to(device)

#                 # U_c = u(system.z_c)
#                 # res = G(U_c)
#                 # pde_loss = loss_fn(bvp.f(u, params), res) if system.z_c.numel() > 0 else torch.tensor(0).to(device) # No collocation points

#                 # Total loss
#                 cur_losses = torch.stack([bc_loss, pde_loss, data_loss])
#                 lambda_ = SoftAdapt(cur_losses, prev_losses, beta=beta_softadapt, loss_weigthed=True)   # SoftAdapt weights
#                 loss = torch.dot(lambda_, cur_losses)

#                 # Log losses to wandb
#                 if log_wandb is not None:
#                     wandb.log({
#                         "Epoch": epoch,
#                         "LR": scheduler_.get_last_lr()[0] if scheduler_ is not None else optimizer_.param_groups[0]['lr'],
#                         "Loss": loss.item(),
#                         "BC Loss": bc_loss.item(),
#                         "PDE Loss": pde_loss.item(),
#                         "Data Loss": data_loss.item(),
#                         **{key: value.item() for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
#                         "lambda_bc": lambda_[0].item(),
#                         "lambda_pde": lambda_[1].item(),
#                         "lambda_data": lambda_[2].item(),
#                     })

#                 # Backpropagate
#                 loss.backward(retain_graph=True)

#                 # Refine boundary and collocation points
#                 boundary_points = boundary_refiner(boundary_points, bc_loss)
#                 collocation_points = collocation_refiner(collocation_points, pde_loss)

#                 return loss, cur_losses.clone()

#             loss, prev_losses = optimizer_.step(closure)  # Pass the closure to the optimizer
#             scheduler_.step(loss.item()) if scheduler_ is not None else None


#             # Plot solution
#             if log_wandb is not None:
#                 if log_wandb["plotter"] is not None and epoch % log_wandb["plot_interval"] == 0:
#                     with torch.no_grad():
#                         plots = log_wandb["plotter"](u, G)
#                         for name, fig in plots.items():
#                             wandb.log({name: wandb.Plotly(fig)})
            

#             if loss_tol is not None:
#                 if loss.item() < loss_tol:
#                     print(f"Loss below tolerance at epoch {epoch}. Terminating training.")


#         # Save the model
#         if save_model["save"]:
#             print("Saving model...")
#             torch.save(u.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_u.pth'))
#             torch.save(G.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_G.pth'))

            