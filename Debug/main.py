# Torch
import torch

# Custom imports
import sys
import os

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"


# Set random seed for reproducibility
torch.manual_seed(42)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, zeros_
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# FEEDFORWARD NEURAL NETWORK
class FNN(nn.Module):
    def __init__(self, dims, hidden_act=nn.Tanh(), output_act=nn.Identity(), weight_init=xavier_normal_, bias_init=zeros_, scale_fn=lambda x: x):

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
    


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )










from scipy.integrate import odeint

class LotkaVolterra:
    def __init__(self, alpha, beta, gamma, delta, X0):
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.X0 = X0

    def f(self, X, t, alpha, beta, gamma, delta):
        x, y = X
        dxdt = alpha*x - beta*x*y
        dydt = -delta*y + gamma*x*y
        return [dxdt, dydt]
    
    def solve(self, t):
        out = odeint(self.f, self.X0, t, (self.alpha, self.beta, self.gamma, self.delta))
        return torch.tensor(out, dtype=torch.float32)




def to_numpy(tensor):
    # Detach tensor from gradient computation and move to CPU as numpy array
    if type(tensor) != torch.Tensor:
        return tensor
    return tensor.squeeze().detach().cpu().numpy()


def sample_with_noise(N, t, X, epsilon=5e-3):

    # Check if the shapes are correct and feasible amount of points are requested
    assert len(X) != len(t) or N <= len(t), "Invalid shapes or N"

    # Calculate the mean of the data
    X_bar = torch.mean(X, dim=0)

    # Sample N evenly spaced points from the data
    idx = torch.linspace(0, len(t)-1, N, dtype=torch.int)
    t, X = t[idx], X[idx]

    # Add noise to the data
    X_noise = X + epsilon * X_bar * torch.randn(*X.shape)

    return t, X_noise


def SoftAdapt(cur_losses, prev_losses, beta=0, loss_weigthed=False):
    f = cur_losses.detach()
    fm1 = prev_losses.detach()
    s = f - fm1
    if loss_weigthed:
        return f*torch.exp(beta*s) / torch.sum(f*torch.exp(beta*s))
    else:
        return torch.nn.functional.softmax(beta*s, dim=0)




import wandb

from abc import abstractmethod
import torch

class BVP:

    def __init__(self, params):
        """
        Boundary Value Problem object: Can evaluate the PDE residual and boundary residual.
        params: dict
            Parameter values of the BVP. If some parameters are to be estimated, they should be torch.nn.Parameter. As an example:
            params = {
                alpha: torch.nn.Parameter(torch.tensor(1.0)),
                beta: 2.3,
                }
            This will estimate alpha from an initial guess of 1.0, while beta will be kept constant at 2.3.
        """
        self.params = params

    def to(self, device):
        # Move all attributes to device
        for attr in self.__dict__.keys():
            if torch.is_tensor(getattr(self, attr)):
                setattr(self, attr, getattr(self, attr).to(device))

    @abstractmethod
    def f(self, z, U):
        """
        z: torch.Tensor
            Input points where the PDE residual is to be evaluated.

        U: torch.Tensor
            Solution (or estimate) of the PDE at the input points z. Should be the output of the solution network u(z).

        params: dict
            Parameter values of the BVP.
        """

        # This method must be implemented
        # It should return the PDE residual
        pass

    @abstractmethod
    def g(self, z, U):
        """
        z: torch.Tensor
            Input points where the boundary residual is to be evaluated.

        U: torch.Tensor
            Solution (or estimate) of the PDE at the input points z. Should be the output of the solution network u(z).

        params: dict
            Parameter values of the BVP.
        """
        # This method must be implemented
        # It should return the boundary residual
        # If there are both Dirichlet and Neumann boundary conditions, stack them on top of each other
        pass


#####################
### Training loop ###
#####################



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
            boundary_refiner: callable = lambda z, loss: z,
            collocation_refiner: callable = lambda z, loss: z,
            loss_fn: torch.nn.Module = torch.nn.MSELoss(),
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_args: dict = dict(lr=1e-3, weight_decay=0.0),
            scheduler: torch.optim.Optimizer = None,
            scheduler_args: dict = None,
            epochs: int = 1000,
            loss_tol: float = None,    # Stop training if loss is below this value
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            beta_softadapt: float = 0.1,
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

        # Initialize previous losses for SoftAdapt
        prev_losses = torch.zeros(3).to(device)

        for epoch in range(epochs):

            def closure():
                optimizer.zero_grad()

                # Boundary condition loss
                bc_residual = bvp.g(boundary_points, u(boundary_points)) if boundary_points is not None else torch.tensor(0).to(device) # No boundary conditions
                bc_loss = loss_fn(bc_residual, torch.zeros_like(bc_residual)) # Enforce boundary conditions

                # Data loss
                data_loss = loss_fn(u(data_points), data_target) if data_points is not None else torch.tensor(0).to(device) # No data

                # PDE loss
                U_c = u(collocation_points)
                res = G(U_c)
                pde_loss = loss_fn(bvp.f(collocation_points, u(collocation_points)), res)

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
                        **{key: value.item() for key, value in [param for param in bvp.params.items() if isinstance(param[1], torch.nn.Parameter)]},
                        "lambda_bc": lambda_[0].item(),
                        "lambda_pde": lambda_[1].item(),
                        "lambda_data": lambda_[2].item(),
                    })

                # Backpropagate
                loss.backward(retain_graph=True)

                return loss, bc_loss, pde_loss, data_loss, cur_losses.clone()


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















class LV_BVP(BVP):
    
    def __init__(self, params):
        super().__init__(params)

    def f(self, z, U):
        alpha = self.params['alpha'] if 'alpha' in self.params else self.alpha
        beta = self.params['beta'] if 'beta' in self.params else self.beta
        delta = self.params['delta'] if 'delta' in self.params else self.delta

        dUdt = torch.cat([
        torch.autograd.grad(outputs=U[:, i], inputs=z, grad_outputs=torch.ones_like(U[:, i]), create_graph=True)[0]
        for i in range(U.shape[1])
        ], dim=-1)

        return torch.stack([
            dUdt[:, 0] - alpha*U[:, 0] + beta*U[:, 0]*U[:, 1],
            dUdt[:, 1] + delta*U[:, 1] # - gamma*U[:, 0]*U[:, 1] <-- Estimate this
        ], dim=-1)
    

    def g(self, z, U):
        return U - torch.tensor([1.0, 1.0], device=U.device)








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
t_d, X_d = sample_with_noise(10, t[train_idx], X[train_idx], epsilon=5e-3)





# Define model architectures
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

# Define model
# params = dict(
#     alpha=torch.nn.Parameter(torch.tensor(0.5)),
#     beta=torch.nn.Parameter(torch.tensor(0.5)),
#     delta=torch.nn.Parameter(torch.tensor(0.5))
# )
params = dict(
    alpha=alpha,
    beta=beta,
    delta=delta,
)



import plotly.graph_objects as go

class LV_Plotter:
    def __init__(self, t, X, t_d, X_d, gamma):
        self.t = t
        self.X = X
        self.z_d = t_d
        self.U_d = X_d
        self.gamma = gamma

    def __call__(self, u, G):
        device = next(u.parameters()).device # Get the device of the model
        plots = dict()

        # Evaluate the model
        X_pred = u(self.t.unsqueeze(-1).to(device))

        # Plot with plotly
        fig = go.Figure()
        fig.add_scatter(x=self.t.cpu().numpy(), y=self.X[:, 0].cpu().numpy(), mode='lines', name='Prey', line=dict(dash='dash', color='green'))
        fig.add_scatter(x=self.t.cpu().numpy(), y=self.X[:, 1].cpu().numpy(), mode='lines', name='Predator', line=dict(dash='dash', color='red'))
        fig.add_scatter(x=self.t.cpu().numpy(), y=X_pred[:, 0].cpu().numpy(), mode='lines', name='Prey (pred)', line=dict(color='green'))
        fig.add_scatter(x=self.t.cpu().numpy(), y=X_pred[:, 1].cpu().numpy(), mode='lines', name='Predator (pred)', line=dict(color='red'))
        # Add datapoints
        fig.add_scatter(x=self.z_d.squeeze().cpu().numpy(), y=self.U_d[:, 0].cpu().numpy(), mode='markers', name='Prey (data)', marker=dict(color='green', symbol='x'))
        fig.add_scatter(x=self.z_d.squeeze().cpu().numpy(), y=self.U_d[:, 1].cpu().numpy(), mode='markers', name='Predator (data)', marker=dict(color='red', symbol='x'))
        fig.update_layout(title=f"Lotka-Volterra Model")
        
        # Log figure to wandb
        plots["Solution"] = fig

        # Plot missing terms
        res = G(X_pred).cpu()
        res_dx = res[:, 0]
        res_dy = res[:, 1]
        true_res_dx = torch.zeros_like(res_dx)
        true_res_dy = (self.gamma*self.X[:, 0]*self.X[:, 1]).cpu().numpy()

        fig = go.Figure()
        fig.add_scatter(x=self.t.cpu().numpy(), y=res_dx, mode='lines', name='Residual Prey', line=dict(color='green'))
        fig.add_scatter(x=self.t.cpu().numpy(), y=res_dy, mode='lines', name='Residual Predator', line=dict(color='red'))
        fig.add_scatter(x=self.t.cpu().numpy(), y=true_res_dx, mode='lines', name='Prey: 0', line=dict(dash='dash', color='green'))
        fig.add_scatter(x=self.t.cpu().numpy(), y=true_res_dy, mode='lines', name='Predator: γ*x*y', line=dict(dash='dash', color='red'))
        fig.update_layout(title=f"Lotka-Volterra Missing Terms")

        # Log figure to wandb
        plots["Missing Terms"] = fig

        return plots

plotter = LV_Plotter(t, X, t_d, X_d, gamma=gamma)



# bvp = LV_BVP(dict(alpha=alpha, beta=torch.nn.Parameter(torch.tensor(1.0)), delta=delta))
bvp = LV_BVP(dict(alpha=alpha, beta=beta, delta=delta))


# Train the model

upinn = UPINN(u, G, bvp)


# def train(self,
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


upinn.train(
    data_points=t_d.unsqueeze(-1),
    data_target=X_d,
    boundary_points=torch.tensor([[0.0]]),
    collocation_points=t[train_idx].unsqueeze(-1).requires_grad_(True),
    epochs=10000,
    log_wandb=dict(name='UPINN', project='Master-Thesis', plotter=plotter, plot_interval=1000),
    optimizer=torch.optim.AdamW,
    optimizer_args=dict(lr=1e-2, weight_decay=1e-10),
    beta_softadapt=0.1,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_args=dict(factor=0.5, patience=100, min_lr=1e-6),
    loss_tol=1e-5,
    )
