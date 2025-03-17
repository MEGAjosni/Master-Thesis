# Torch
import torch

# Custom imports
import os
from tqdm import tqdm
from itertools import count
import traceback
import sys
sys.path.append('../utils')
from utils import SoftAdapt, RAD_sampler, RAR_sampler, sample_collocation_points

# Weights and Biases
import wandb
os.environ["WANDB_MODE"] = "online" # Only log to WandB online
import  tempfile
folder_temp = tempfile.TemporaryDirectory()
os.chmod(folder_temp.name, 0o777)


class NullWork(torch.nn.Module):
    """ A class to represent absence of Network or Operator """
    def __init__(self, out_dim=1):
        super(NullWork, self).__init__()
        self.out_dim = out_dim
    
    def forward(self, x):
        return torch.zeros(x.shape[0], self.out_dim, device=x.device)


class UPINN(torch.nn.Module):

    def __init__(self,
        u: torch.nn.Module,
        N: torch.nn.Module = None,
        F: torch.nn.Module = None,
        boundary_points: tuple = None,
        data_points: tuple = None,
        collocation_points: torch.Tensor = None,
        inductive_bias: bool = True,
    ):
        """
        A class to train Universal Physics-Informed Neural Networks (UPINNs) for solving PDEs.
        
        The class assumes the PDE to be of the form,
            x = (x1, x2, ..., xn), u = u(x)
            N(x, u) + F(x, u) = 0
        where N is the known dynamics of the system and F is the unknown dynamics.

        u: torch.nn.Module (Trainable)
            Surrogate: Neural Network estimating the solution of the PDE

        N: torch.nn.Module (May contain trainable parameters or unknown terms modelled by trainable Neural Networks e.g. source, forcing, etc)
            Operator: Compute the known dynamics of the system

        F: torch.nn.Module (Trainable)
            Operator: Neural Network estimating the unknown dynamics of the system

        data_points: tuple of 2 torch.Tensor
            Tuple of torch.Tensor containing the data points (X, U)
            Default: Tuple of empty tensors

        boundary_points: tuple of 2 torch.Tensor
            Tuple of torch.Tensor containing the boundary points (X, U)
            Default: Tuple of empty tensors
        
        collocation_points: torch.Tensor
            Torch.Tensor containing the collocation points X
            Default: Empty tensor
        """
    
        super(UPINN, self).__init__()

        # Torch modules
        self.u = u
        if N is None:
            print('[Info]: Initializing NN model (Known dynamics N unspecified; Setting N and F to 0.0)')
            self.N = NullWork(); self.F = NullWork()
        elif F is None:
            print('[Info]: Initializing PINN model (Residual network F unspecified; Setting F to 0.0)')
            self.N = N; self.F = NullWork()
        else:
            print('[Info]: Initializing UPINN model')
            self.N, self.F = N, F
        self.device = torch.device('cpu')

        # Training data
        # empty_input = torch.empty(0, self.u.layers[0].in_features)
        # empty_output = torch.empty(0, self.u.layers[-1].out_features)
        # self.data_points = (empty_input, empty_output) if data_points is None else data_points
        # self.boundary_points = (empty_input, empty_output) if boundary_points is None else boundary_points
        # self.collocation_points = empty_input if collocation_points is None else collocation_points
        # self.collocation_points.requires_grad = True
        self.data_points = data_points; self.N_data = len(data_points[0]) if data_points is not None else None
        self.boundary_points = boundary_points; self.N_boundary = len(boundary_points[0]) if boundary_points is not None else None
        self.collocation_points = collocation_points; self.N_coll = len(collocation_points) if collocation_points is not None else None
        if self.collocation_points is not None: self.collocation_points.requires_grad = True

        # Training
        self.optimizer = torch.optim.AdamW([*self.u.parameters(), *self.F.parameters(), *self.N.parameters()], lr=1e-3)
        self.scheduler = None
        self.softadapt_kwargs = dict()
        self.lambdas = torch.tensor([1.0, 1.0, 1.0])
        self.softadapt_interval = 50
        self.inductive_bias = inductive_bias

        # Logging
        self.log = dict()
        self.log_to_wandb = False
        self.epoch = 0


    # Move model and data to device
    def to(self, device):
        self.device = device
        super(UPINN, self).to(device)
        self.data_points = (self.data_points[0].to(device), self.data_points[1].to(device)) if self.data_points is not None else None
        self.boundary_points = (self.boundary_points[0].to(device), self.boundary_points[1].to(device)) if self.boundary_points is not None else None
        self.collocation_points = self.collocation_points.to(device) if self.collocation_points is not None else None
    
    # Save model
    def save(self, name, path='', overwrite=False, save_architecture=True, save_individual_models=False):

        # Ensure save directory exists
        os.makedirs(path, exist_ok=True) if path else None
        save_path = os.path.join(path, name)

        # Find available filename
        extensions = ['_u.pt', '_N.pt', '_F.pt'] if save_individual_models else ['.pt']
        if overwrite: suffix = ''
        else:
            name_not_available = lambda suffix: any(os.path.exists(save_path + suffix + ext) for ext in extensions)
            suffix = '_' + str(next((i for i in count(1) if not name_not_available(str(i))), '')) if name_not_available('') else ''
            if suffix: print(f"[Info]: {name} already exists in save directory. Enumerating model as {suffix}. To overwrite, set overwrite=True.")

        # Save the model
        try:
            for key in ['u', 'N', 'F']: torch.save(getattr(self, key).state_dict(), f"{save_path}{suffix}_{key}.pt") if save_individual_models else torch.save(self.state_dict(), f"{save_path}{suffix}.pt")
            with open(f"{save_path}{suffix}_architecture.txt", 'w') as f: f.write(f"u: {self.u}\nN: {self.N}\nF: {self.F}") if save_architecture else None
            print(f"[Info]: Successfully saved {'models individually' if save_individual_models else 'total model'} with name {name}{suffix} at {path if path else 'current directory'}")
            
        except FileNotFoundError: print("[Error]: Failed to save model.")
    
    
    # Load model
    def load(self, name, path='', load_individual_models=False):

        # Remove file extension if present
        name = name.split('.')[0]

        # Check if model exists
        exists = any(os.path.exists(os.path.join(path, f"{name}_{key}.pt")) for key in ['u', 'N', 'F']) if load_individual_models else os.path.exists(os.path.join(path, f"{name}.pt"))
        if not exists: print("[Error]: Model not found."); return
        
        # Load the model
        try:
            if load_individual_models:
                for key in ['u', 'N', 'F']: getattr(self, key).load_state_dict(torch.load(os.path.join(path, f"{name}_{key}.pt"), weights_only=True))
                print(f"[Info]: Model with name {name} loaded successfully from {path or 'current directory'}")
            else:
                params = torch.load(os.path.join(path, f"{name}.pt"), weights_only=True)
                try: self.load_state_dict(params)
                except:
                    # If failed to load, try loading individual models
                    for key in ['u', 'N', 'F']:
                        try: getattr(self, key).load_state_dict({k[len(key)+1:]: v for k, v in params.items() if k.startswith(key)})
                        except:
                            print(f"[Error]: Failed to load {key} model. Check compatibility with {name}_architecture.txt")
                            # Print out the error message that triggered the exception
                            traceback.print_exc()

        except:
            print("[Error]: Failed to load model.")


    def freeze(self, model='all'):
        if model == 'all':
            self.requires_grad_(False)
        elif model == 'u':
            self.u.requires_grad_(False)
        elif model == 'N':
            self.N.requires_grad_(False)
        elif model == 'F':
            self.F.requires_grad_(False)
        else:
            raise ValueError(f"Invalid model: {model}. Choose from ['all', 'u', 'N', 'F']")
    
    def unfreeze(self, model='all'):
        if model == 'all':
            self.requires_grad_(True)
        elif model == 'u':
            self.u.requires_grad_(True)
        elif model == 'N':
            self.N.requires_grad_(True)
        elif model == 'F':
            self.F.requires_grad_(True)
        else:
            raise ValueError(f"Invalid model: {model}. Choose from ['all', 'u', 'N', 'F']")


    # Forward pass
    def predict_solution(self, X):
        return self.u(X)
    
    def predict_residual(self, X):
        return self.F(self.F_input(X, self.u(X)))
        
    def evaluate_known(self, X):
        return self.N(X, self.u(X))


    # Adaptive refinement of collocation points
    def refine_collocation_points(self, method, lb, ub, N_new=None, N_candidates=None, N_max=None, method_kwargs=dict(), sample_method='sobol'):

        if N_new is None: N_new = self.N_coll
        if N_candidates is None: N_candidates = 50*N_new
        if N_max is None: N_max = 5*self.N_coll # Maximum number of allowed collocation points

        if N_new > N_candidates: raise ValueError("N_new should be less than or equal to N_candidates.")

        # Sample candidate points
        Xc = sample_collocation_points(N_candidates, len(lb), lb=lb, ub=ub, method=sample_method).requires_grad_(True).float()

        # Compute the residual
        u = self.u(Xc)
        pde_loss = abs(self.N(Xc, u) + self.F(self.F_input(Xc, u)))
        residuals = torch.sum(pde_loss, dim=1)

        if method == 'RAD':
            # Replaces all collocation points with new points
            if method_kwargs == {}: method_kwargs = dict(k=0.5, c=0.1) # Use recommended default values from: Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). https://doi.org/10.1016/J.CMA.2022.115671
            self.collocation_points = RAD_sampler(Xc, residuals, N_new, **method_kwargs)
        
        elif method == 'RAR':
            # Appends new points to the existing collocation points
            if len(self.collocation_points) < N_max:
                self.collocation_points = torch.cat([self.collocation_points, RAR_sampler(Xc, residuals, N_new, **method_kwargs)])

        elif method == 'RAR-D':
            # Appends new points to the existing collocation points
            if len(self.collocation_points) < N_max:
                self.collocation_points = torch.cat([self.collocation_points, RAD_sampler(Xc, residuals, N_new, **method_kwargs)])

        else:
            raise ValueError(f"Invalid method: {method}. Choose from ['RAD', 'RAR', 'RAR-D']")
        

    # Default loss functions
    def bc_loss(self):
        if self.boundary_points is not None: 
            U_bc = self.u(self.boundary_points[0])
            bc_loss = torch.nn.MSELoss()(U_bc, self.boundary_points[1])
        else: bc_loss = torch.tensor(0.0)
        return bc_loss

    def data_loss(self):
        if self.data_points is not None:
            U_data = self.u(self.data_points[0]) 
            data_loss = torch.nn.MSELoss()(U_data, self.data_points[1])
        else: data_loss = torch.tensor(0.0)
        return data_loss

    def F_input(self, Z, U):
        return torch.cat([Z, U], dim=1)

    def pde_loss(self):
        if self.collocation_points is not None:
            U_c = self.u(self.collocation_points)
            res = self.F(self.F_input(self.collocation_points, U_c))
            known = self.N(self.collocation_points, U_c)
            pde_loss = torch.nn.MSELoss()(known, -res) if self.collocation_points.shape[0] > 0 else torch.tensor(0.0)
        else: pde_loss = torch.tensor(0.0)
        return pde_loss
    
    def get_loss(self):
        bc_loss = self.bc_loss()
        data_loss = self.data_loss()
        pde_loss = self.pde_loss()
        if self.epoch % self.softadapt_interval == 0 and len(self.softadapt_kwargs) > 0:
            self.lambdas = SoftAdapt(**self.softadapt_kwargs)(torch.tensor([bc_loss, data_loss, pde_loss]))
        loss = self.lambdas[0]*bc_loss + self.lambdas[1]*data_loss + self.lambdas[2]*pde_loss

        return loss, bc_loss, data_loss, pde_loss


    # Default closure: Can be modified
    def closure(self):

        # Zero the gradients
        self.optimizer.zero_grad()

        # Compute loss
        loss, bc_loss, data_loss, pde_loss = self.get_loss()

        # Backpropagate
        loss.backward(retain_graph=True)

        # Log the losses
        self.log.setdefault('bc_loss', []).append(bc_loss.item())
        self.log.setdefault('data_loss', []).append(data_loss.item())
        self.log.setdefault('pde_loss', []).append(pde_loss.item())
        self.log.setdefault('loss', []).append(loss.item())

        log = {k: v[-1] for k, v in self.log.items()}

        # Log to WandB or tqdm
        wandb.log(log) if self.log_to_wandb else self.epoch_iterator.set_postfix({k: f"{v:.2e}" for k, v in log.items()})

        return loss
    
    # Train the model
    def train_loop(
            self,
            epochs: int = 1000,
            optimizer: torch.optim.Optimizer = None,
            scheduler: torch.optim.Optimizer = None,
            loss_tol: float = -torch.inf,    # Stop training if loss is below this value
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            log_wandb_kwargs: dict = dict(name='UPINN', project='Master-Thesis'),
    ):

        # Move the model and data to the device
        self.train()
        self.to(device); self.device = device
        
        # Setup optimizer
        # if optimizer: self.optimizer = optimizer([*self.u.parameters(), *self.F.parameters(), *self.N.parameters()], **optimizer_kwargs)
        if optimizer: self.optimizer = optimizer
        print(f"[Info]: Training {epochs} epoch(s) on {self.device} using {self.optimizer.__class__.__name__} optimizer.")

        # Setup learning rate scheduler
        # if scheduler: self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        if scheduler: self.scheduler = scheduler

        # Setup monitoring - Either WandB (online-dashboard) or tqdm (local-terminal)
        if self.log_to_wandb:
            wandb.init(**log_wandb_kwargs)
            wandb.watch([self.u, self.N, self.F])
            self.epoch_iterator = range(epochs)
        else:
            self.epoch_iterator = tqdm(range(epochs), desc=f"[{str(device).upper()}]", unit=' epoch')


        # Training loop
        for epoch in self.epoch_iterator:
            self.epoch += 1

            # Take optimization step
            loss = self.optimizer.step(self.closure)
            if self.scheduler: self.scheduler.step(loss)
            
            # Stopping criterions
            if loss.item() < loss_tol: print(f"[Info]: Loss below tolerance at epoch {epoch}. Terminating training."); break
            if torch.isnan(loss): raise RuntimeError(f"Loss became NaN at epoch {epoch}/{epochs}. Terminating training.")
        

        # print("[Info]: Training complete.")
        self.eval()
        self.to(torch.device('cpu')); self.device = torch.device('cpu')
    
    def plot():
        raise NotImplementedError("Method not implemented.")
    