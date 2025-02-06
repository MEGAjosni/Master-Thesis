# Torch
import torch

# Custom imports
import os
from tqdm import tqdm
from itertools import count

# Weights and Biases
import wandb
os.environ["WANDB_MODE"] = "online" # Only log to WandB online
import  tempfile
folder_temp = tempfile.TemporaryDirectory()
os.chmod(folder_temp.name, 0o777)


class NullWork(torch.nn.Module):
    """ A class to represent absence of Network or Operator """
    def __init__(self):
        super(NullWork, self).__init__()
    def forward(self, x):
        return torch.tensor(0.0)


class UPINN:

    def __init__(self,
        u: torch.nn.Module,
        F: torch.nn.Module = None,
        G: torch.nn.Module = None,
        data_points: tuple = None,
        initial_points: tuple = None,
        boundary_points: tuple = None,
        collocation_points: torch.Tensor = None,
    ):
        """
        A class to train Universal Physics-Informed Neural Networks (UPINNs) for solving PDEs.
        
        The class assumes the PDE to be of the form,
            x = (t, x1, x2, ..., xn), u = u(x)
            F(x, u) + G(x, u) = 0
        where F is the known dynamics of the system and G is the unknown dynamics.

        u: torch.nn.Module (Trainable)
            Surrogate: Neural Network estimating the solution of the PDE

        F: torch.nn.Module (May contain trainable parameters or unknown terms modelled by trainable Neural Networks e.g. source, forcing, etc)
            Operator: Compute the known dynamics of the system

        G: torch.nn.Module (Trainable)
            Operator: Neural Network estimating the unknown dynamics of the system

        data_points: tuple of 2 torch.Tensor
            Tuple of torch.Tensor containing the data points (X, U)
            Default: Tuple of empty tensors
        
        initial_points: tuple of 2 torch.Tensor
            Tuple of torch.Tensor containing the initial points (X, U)
            Default: Tuple of empty tensors

        boundary_points: tuple of 2 torch.Tensor
            Tuple of torch.Tensor containing the boundary points (X, U)
            Default: Tuple of empty tensors
        
        collocation_points: torch.Tensor
            Torch.Tensor containing the collocation points X
            Default: Empty tensor
        """
    
        # Torch modules
        self.u = u
        if F is None:
            print('[Info]: Initializing NN model (Known dynamics F unspecified; Setting F and G to 0.0)')
            self.F = NullWork(); self.G = NullWork()
        elif G is None:
            print('[Info]: Initializing PINN model (Residual network G unspecified; Setting G to 0.0)')
            self.F = F; self.G = NullWork()
        else:
            print('[Info]: Initializing UPINN model')
            self.F, self.G = F, G

        # Training data
        empty_input = torch.empty(0, self.u.layers[0].in_features)
        empty_output = torch.empty(0, self.u.layers[-1].out_features)
        self.data_points = (empty_input, empty_output) if data_points is None else data_points
        self.initial_points = (empty_input, empty_output) if initial_points is None else initial_points
        self.boundary_points = (empty_input, empty_output) if boundary_points is None else boundary_points
        self.collocation_points = empty_input if collocation_points is None else collocation_points
        self.collocation_points.requires_grad = True

        # Logging
        self.log = dict()
        self.log_to_wandb = False


    # Move model and data to device
    def to(self, device):
        self.u.to(device)
        self.F.to(device)
        self.G.to(device)
        self.data_points = (self.data_points[0].to(device), self.data_points[1].to(device))
        self.initial_points = (self.initial_points[0].to(device), self.initial_points[1].to(device))
        self.boundary_points = (self.boundary_points[0].to(device), self.boundary_points[1].to(device))
        self.collocation_points = self.collocation_points.to(device)
    
    # Set model mode
    def train(self): self.u.train(); self.F.train(); self.G.train()
    def eval(self): self.u.eval(); self.F.eval(); self.G.eval()
    
    # Save model
    def save(self, name, path=''):

        # Ensure save directory exists
        os.makedirs(path, exist_ok=True) if path else None
        save_path = os.path.join(path, name)

        # Find available filename
        name_not_available = lambda suffix: any(os.path.exists(save_path + suffix + ext) for ext in ['_u.pth', '_F.pth', '_G.pth'])
        suffix = str(next((i for i in count(1) if not name_not_available(str(i))), '')) if name_not_available('') else ''
        if suffix: print(f"[Info]: Model name already exists in save directory. Enumerating model as {suffix}")

        # Save the model
        try:
            for key in ['u', 'F', 'G']: torch.save(getattr(self, key).state_dict(), f"{save_path}{suffix}_{key}.pth")
            with open(f"{save_path}{suffix}_architecture.txt", 'w') as f: f.write(f"u: {self.u}\nF: {self.F}\nG: {self.G}")
            print(f"[Info]: Model saved successfully with name {name}{suffix} at {path if path else 'current directory'}")
            
        except FileNotFoundError:
            print("[Error]: Failed to save model.")
    
    
    # Load model
    def load(self, name, path=''):

        # Check if model exists
        if not all(os.path.exists(os.path.join(path, f"{name}_{key}.pth")) for key in ['u', 'F', 'G']):
            print("[Error]: Model not found.")
            return
        
        # Load the model
        try:
            for key in ['u', 'F', 'G']:
                getattr(self, key).load_state_dict(torch.load(os.path.join(path, f"{name}_{key}.pth"), weights_only=True))
            print(f"[Info]: Model loaded successfully from {path if path else 'current directory'}")
        except FileNotFoundError:
            print("[Error]: Failed to load model.")


    # Forward pass
    def predict(self, X):
        with torch.no_grad():
            return self.u(X)
    
    def predict_residual(self, X):
        with torch.no_grad():
            return self.G(self.u(X))
        
    def evaluate_known_dynamics(self, X):
        with torch.no_grad():
            return self.F(X, self.u(X))

    # Functions to be implemented by the user if needed
    def refine_initial_points(self): pass
    def refine_boundary_points(self): pass
    def refine_collocation_points(self): pass

    # Default loss functions
    def init_loss(self):
        U_init = self.u(self.initial_points[0])
        init_loss = torch.nn.MSELoss()(U_init, self.initial_points[1]) if self.initial_points[0].shape[0] > 0 else torch.tensor(0.0)
        return init_loss

    def bc_loss(self):
        U_bc = self.u(self.boundary_points[0])
        bc_loss = torch.nn.MSELoss()(U_bc, self.boundary_points[1]) if self.boundary_points[0].shape[0] > 0 else torch.tensor(0.0)
        return bc_loss

    def data_loss(self):
        U_data = self.u(self.data_points[0])
        data_loss = torch.nn.MSELoss()(U_data, self.data_points[1]) if self.data_points[0].shape[0] > 0 else torch.tensor(0.0)
        return data_loss

    def pde_loss(self):
        U_c = self.u(self.collocation_points)
        res = self.G(torch.cat([self.collocation_points, U_c], dim=1))
        pde_loss = torch.nn.MSELoss()(self.F(self.collocation_points, U_c), res) if self.collocation_points.shape[0] > 0 else torch.tensor(0.0)
        return pde_loss

    # Default closure: Can be modified
    def closure(self):

        # Total loss
        init_loss, bc_loss, data_loss, pde_loss = self.init_loss(), self.bc_loss(), self.data_loss(), self.pde_loss()
        loss = init_loss + bc_loss + data_loss + pde_loss

        # Log
        self.log.update({
            "L": loss.item(),
            "L_ic": init_loss.item(),
            "L_bc": bc_loss.item(),
            "L_data": data_loss.item(),
            "L_pde": pde_loss.item()
        })

        if self.log_to_wandb:
            wandb.log(self.log)
        else:
            # Log in scientific notation to 6 decimal places
            self.epoch_iterator.set_postfix({k: f"{v:.2e}" for k, v in self.log.items()})

        # Refine points
        self.refine_initial_points()
        self.refine_boundary_points()
        self.refine_collocation_points()

        return loss
    
    # Train the model
    def train_loop(
            self,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_args: dict = dict(lr=1e-3),
            scheduler: torch.optim.Optimizer = None,
            scheduler_args: dict = dict(),
            epochs: int = 1000,
            loss_tol: float = -torch.inf,    # Stop training if loss is below this value
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            save_name: dict = None,
            save_path = '',
            log_wandb: dict = dict(name='UPINN', project='Master-Thesis', plotter=None, plot_interval=1000)
    ):

        print("[Info]: Beginning training...")

        # Move the model and data to the device
        self.train()
        self.to(device)
        
        # Setup optimizer
        optimizer = optimizer([
            {'params': self.u.parameters()},
            {'params': self.F.parameters()},
            {'params': self.G.parameters()}
        ], **optimizer_args)
             
        # Setup learning rate scheduler
        if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_args)

        # Setup monitoring - Either WandB (online-dashboard) or tqdm (local-terminal)
        if log_wandb is not None and self.log_to_wandb:
            wandb.init(project=log_wandb["project"], name=log_wandb["name"])
            wandb.watch(self.u)
            wandb.watch(self.F)
            wandb.watch(self.G)
            self.epoch_iterator = range(epochs)
        else:
            self.epoch_iterator = tqdm(range(epochs), desc=f"[{str(device).upper()}]", unit=' epoch')

        # Training loop
        for epoch in self.epoch_iterator:
            
            # Zero the gradients
            optimizer.zero_grad()

            # Compute loss
            loss = self.closure()

            # Backpropagate
            loss.backward(retain_graph=True)

            # Take optimization step
            optimizer.step()

            # Stopping criterion
            if loss.item() < loss_tol:
                print(f"[Info]: Loss below tolerance at epoch {epoch}. Terminating training.")
                break
            if torch.isnan(loss):
                raise RuntimeError(f"Loss became NaN at epoch {epoch}/{epochs}. Terminating training.")
        
        print("[Info]: Training complete.")
        print("[Info]: Moving model to CPU...")
        self.eval()
        self.to(torch.device('cpu'))

        if save_name: self.save(save_name, save_path)


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
    
#     def to(self, device):
#         self.u.to(device)
#         self.G.to(device)
#         self.bvp.to(device)
        

#     def train(
#             self,
#             optimizer: torch.optim.Optimizer,
            
            
#             data_points: torch.Tensor = None,
#             data_target: torch.Tensor = None,
#             boundary_points: torch.Tensor = None,
#             collocation_points: torch.Tensor = None,
#             lambda_reg: float = 0.0,
#             priotize_pde: float = 1.0,
#             boundary_refiner: callable = lambda z, loss: z,
#             collocation_refiner: callable = lambda z, loss: z,
#             loss_fn_bc: torch.nn.Module = torch.nn.MSELoss(),
#             loss_fn_data: torch.nn.Module = torch.nn.MSELoss(),
#             loss_fn_pde: torch.nn.Module = torch.nn.MSELoss(),
#             scheduler: torch.optim.Optimizer = None,
#             scheduler_args: dict = None,
#             epochs: int = 1000,
#             loss_tol: float = None,    # Stop training if loss is below this value
#             device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#             beta_softadapt: float = 0.0,
#             save_model: dict = None,
#             log_wandb: dict = dict(name='UPINN', project='Master-Thesis', plotter=None, plot_interval=1000)
#     ):

#         print("Beginning training...")
#         print("Running on:", device)

#         u = self.u; G = self.G; bvp = self.bvp 

#         # Move the model and data to the device
#         u.to(device); G.to(device); bvp.to(device)
#         data_points = data_points.to(device) if data_points is not None else None
#         data_target = data_target.to(device) if data_target is not None else None
#         boundary_points = boundary_points.to(device) if boundary_points is not None else None
#         collocation_points = collocation_points.to(device) if collocation_points is not None else None
        
#         # Initialize optimizer
#         params = [param for param in bvp.params.values() if isinstance(param, torch.nn.Parameter)]
#         optimizer = optimizer([*u.parameters(), *G.parameters(), *params], **optimizer_args)
        
#         # Setup learning rate scheduler
#         if scheduler is not None:
#             scheduler = scheduler(optimizer, **scheduler_args)

#         # Setup WandB logging
#         if log_wandb is not None:
#             wandb.init(project=log_wandb["project"], name=log_wandb["name"])
#             wandb.watch(u)
#             wandb.watch(G)
#             epoch_iterator = range(epochs)
#         else:
#             epoch_iterator = tqdm(range(epochs))

#         # Initialize previous losses for SoftAdapt
#         prev_losses = torch.zeros(3).to(device)
#         loss = 1.0

#         variable_log = dict(loss=None, bc_loss=None, pde_loss=None, data_loss=None, prev_losses=prev_losses)

#         for epoch in epoch_iterator:

#             def closure():
#                 optimizer.zero_grad()

#                 # Boundary condition loss
#                 bc_residual = bvp.g(boundary_points, u(boundary_points)) if boundary_points is not None else torch.tensor(0.0).to(device) # No boundary conditions
#                 bc_loss = loss_fn_bc(bc_residual, torch.zeros_like(bc_residual)) # Enforce boundary conditions

#                 # Data loss
#                 data_loss = loss_fn_data(u(data_points), data_target) if data_points is not None else torch.tensor(0.0).to(device) # No data
#                 # data_loss = torch.mean(torch.mean(DeadZoneLinear(0.05)(torch.abs(u(data_points) - data_target)), dim=0))

#                 # PDE loss
#                 U_c = u(collocation_points)
#                 res = G(U_c)
#                 pde_loss = priotize_pde*loss_fn_pde(bvp.f(collocation_points, U_c), res)

#                 # Regularization loss
#                 reg_loss = torch.mean(torch.abs(res))

#                 # Total loss
#                 cur_losses = torch.stack([bc_loss, pde_loss, data_loss])
#                 lambda_ = SoftAdapt(cur_losses, variable_log["prev_losses"], beta=beta_softadapt)   # SoftAdapt weights
#                 loss = torch.dot(lambda_, cur_losses)

#                 if log_wandb is not None:
#                     wandb.log({
#                         "Epoch": epoch,
#                         "LR": optimizer.param_groups[0]['lr'],
#                         "Loss": loss.item(),
#                         "BC Loss": bc_loss.item(),
#                         "PDE Loss": pde_loss.item(),
#                         "Data Loss": data_loss.item(),
#                         "Reg Loss": reg_loss.item(),
#                         **{key: value.item() for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
#                         "lambda_bc": lambda_[0].item(),
#                         "lambda_pde": lambda_[1].item(),
#                         "lambda_data": lambda_[2].item(),
#                         "lambda_reg": lambda_reg,
#                     })
#                 else:
#                     epoch_iterator.set_postfix({
#                         "Loss": round(loss.item(), 6),
#                         "BC Loss": round(bc_loss.item(), 6),
#                         "PDE Loss": round(pde_loss.item(), 6),
#                         "Data Loss": round(data_loss.item(), 6),
#                         "Reg Loss": round(reg_loss.item(), 6),
#                         "LR_reg": round(lambda_reg, 3),
#                         **{key: round(value.item(), 3) for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
#                     })

#                 # Backpropagate
#                 # nn_params = [param for param in bvp.params.values() if isinstance(param, torch.nn.Parameter)]
#                 # lambda_param = 0.01
#                 # params_loss = torch.sum(torch.nn.ReLU()(-torch.stack(nn_params))) if len(nn_params) > 0 else torch.tensor(0).to(device)
#                 (loss + lambda_reg*reg_loss).backward(retain_graph=True)

#                 variable_log["loss"] = loss.item()
#                 variable_log["bc_loss"] = bc_loss.item()
#                 variable_log["pde_loss"] = pde_loss.item()
#                 variable_log["data_loss"] = data_loss.item()
#                 variable_log["current_losses"] = cur_losses.clone()

#                 return loss

#             # Perform optimization step and update learning rate
#             loss = optimizer.step(closure)  # Pass the closure to the optimizer
#             scheduler.step(loss.item()) if scheduler is not None else None

#             # Refine boundary and collocation points
#             boundary_points = boundary_refiner(boundary_points, variable_log["bc_loss"])
#             collocation_points = collocation_refiner(collocation_points, variable_log["pde_loss"])

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
#                     break
        
#         print("Training complete.")

#         # Save the model
#         if save_model is not None:
#             print("Saving model...")
#             torch.save(u.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_u.pth'))
#             torch.save(G.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_G.pth'))








# # class UPINN:

# #     def __init__(self,
# #         u: torch.nn.Module,
# #         G: torch.nn.Module,
# #         bvp: BVP, # Parameters to be estimated are also included in the BVP object
# #     ):
# #         """
# #         u: torch.nn.Module
# #             Neural Network estimating the solution of the PDE

# #         G: torch.nn.Module
# #             Neural Network estimating the PDE residual

# #         bvp: BVP
# #             Boundary Value Problem object: Can evaluate the PDE residual and boundary residual
# #             Has an attribute for parameters (params: dict)
# #                 Parameters of the BVP.
# #                 If some parameters are to be estimated, they should be torch.nn.Parameter. As an example:
# #                 params = {
# #                     alpha: torch.nn.Parameter(torch.tensor(1.0)),
# #                     beta: 2.3,
# #                     }
# #                 This will estimate alpha from an initial guess of 1.0, while beta will be kept constant at 2.3.
# #         """
    
# #         self.u = u
# #         self.G = G
# #         self.bvp = bvp


# #     def train(self,
# #         data_points: torch.Tensor = None,
# #         data_target: torch.Tensor = None,
# #         boundary_points: torch.Tensor = None,
# #         collocation_points: torch.Tensor = None,
# #         boundary_refiner: callable = lambda z, loss: None,
# #         collocation_refiner: callable = lambda z, loss: None,
# #         loss_fn: torch.nn.Module = torch.nn.MSELoss(),
# #         optimizer: torch.optim.Optimizer = torch.optim.Adam,
# #         optimizer_args: dict = dict(lr=1e-3, weight_decay=0.0),
# #         scheduler: torch.optim.Optimizer = None,
# #         scheduler_args: dict = dict(),
# #         scheduler_extend: torch.optim.Optimizer = None,
# #         scheduler_extend_args: dict = dict(),
# #         epochs: int = 30000,
# #         loss_tol: float = None,    # Stop training if loss is below this value
# #         device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
# #         beta_softadapt: float = 0.1,
# #         save_model: dict = dict(save=False, path='models/lotka-volterra', name='LV-UPINN'),
# #         log_wandb: dict = dict(name='UPINN', project='Master-Thesis', plotter=None, plot_interval=1000)
# #     ):
# #         u = self.u; G = self.G; bvp = self.bvp

# #         print("Beginning training...")
# #         print("Running on:", device)

# #         # Move the model and data to the device
# #         u.to(device); G.to(device); bvp.to(device)
# #         boundary_points = boundary_points.to(device) if boundary_points is not None else None
# #         collocation_points = collocation_points.to(device) if collocation_points is not None else None
# #         data_points = data_points.to(device) if data_points is not None else None
# #         data_target = data_target.to(device) if data_target is not None else None

# #         # Initialize optimizer
# #         optimizer_ = optimizer([*u.parameters(), *G.parameters(), *[param for param in bvp.params.values() if isinstance(param, torch.nn.Parameter)]], **optimizer_args)
        
# #         # Setup learning rate scheduler
# #         scheduler_ = scheduler(optimizer_, **scheduler_args) if scheduler is not None else None

# #         # Setup WandB logging
# #         if log_wandb is not None:
# #             wandb.init(project=log_wandb["project"], name=log_wandb["name"], dir=folder_temp)
# #             wandb.watch(u)
# #             wandb.watch(G)

# #         # Initialize previous losses for SoftAdapt
# #         prev_losses = torch.zeros(3).to(device)

# #         for epoch in range(epochs):

# #             def closure(
# #                     # Make sure the points are in the scope of the closure
# #                     boundary_points=boundary_points,
# #                     collocation_points=collocation_points,
# #                     data_points=data_points,
# #                     data_target=data_target
# #                 ):

# #                 optimizer_.zero_grad()

# #                 # Boundary condition loss
# #                 if boundary_points is not None:
# #                     bc_residual = bvp.g(boundary_points, u(boundary_points))
# #                     bc_loss = loss_fn(bc_residual, torch.zeros_like(bc_residual)) # Enforce boundary conditions
# #                 else:
# #                     bc_loss = torch.tensor(0).to(device) # No boundary conditions
# #                 # bc_loss = loss_fn(u(system.z_b), system.U_b) if system.U_b.numel() > 0 else torch.tensor(0).to(device) # No boundary conditions

# #                 # Data loss
# #                 data_loss = loss_fn(u(data_points), data_target) if data_points is not None else torch.tensor(0).to(device) # No data
# #                 # data_loss = loss_fn(u(system.z_d), system.U_d) if system.U_d.numel() > 0 else torch.tensor(0).to(device) # No data

# #                 # PDE loss
# #                 if collocation_points is not None:
# #                     pde_residual = bvp.f(collocation_points, u(collocation_points))
# #                     pde_loss = loss_fn(pde_residual, G(u(collocation_points))) # Enforce PDE residual
# #                 else:
# #                     pde_loss = torch.tensor(0).to(device)

# #                 # U_c = u(system.z_c)
# #                 # res = G(U_c)
# #                 # pde_loss = loss_fn(bvp.f(u, params), res) if system.z_c.numel() > 0 else torch.tensor(0).to(device) # No collocation points

# #                 # Total loss
# #                 cur_losses = torch.stack([bc_loss, pde_loss, data_loss])
# #                 lambda_ = SoftAdapt(cur_losses, prev_losses, beta=beta_softadapt, loss_weigthed=True)   # SoftAdapt weights
# #                 loss = torch.dot(lambda_, cur_losses)

# #                 # Log losses to wandb
# #                 if log_wandb is not None:
# #                     wandb.log({
# #                         "Epoch": epoch,
# #                         "LR": scheduler_.get_last_lr()[0] if scheduler_ is not None else optimizer_.param_groups[0]['lr'],
# #                         "Loss": loss.item(),
# #                         "BC Loss": bc_loss.item(),
# #                         "PDE Loss": pde_loss.item(),
# #                         "Data Loss": data_loss.item(),
# #                         **{key: value.item() for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
# #                         "lambda_bc": lambda_[0].item(),
# #                         "lambda_pde": lambda_[1].item(),
# #                         "lambda_data": lambda_[2].item(),
# #                     })

# #                 # Backpropagate
# #                 loss.backward(retain_graph=True)

# #                 # Refine boundary and collocation points
# #                 boundary_points = boundary_refiner(boundary_points, bc_loss)
# #                 collocation_points = collocation_refiner(collocation_points, pde_loss)

# #                 return loss, cur_losses.clone()

# #             loss, prev_losses = optimizer_.step(closure)  # Pass the closure to the optimizer
# #             scheduler_.step(loss.item()) if scheduler_ is not None else None


# #             # Plot solution
# #             if log_wandb is not None:
# #                 if log_wandb["plotter"] is not None and epoch % log_wandb["plot_interval"] == 0:
# #                     with torch.no_grad():
# #                         plots = log_wandb["plotter"](u, G)
# #                         for name, fig in plots.items():
# #                             wandb.log({name: wandb.Plotly(fig)})
            

# #             if loss_tol is not None:
# #                 if loss.item() < loss_tol:
# #                     print(f"Loss below tolerance at epoch {epoch}. Terminating training.")


# #         # Save the model
# #         if save_model["save"]:
# #             print("Saving model...")
# #             torch.save(u.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_u.pth'))
# #             torch.save(G.state_dict(), os.path.join(save_model["path"], save_model["name"] + '_G.pth'))

            