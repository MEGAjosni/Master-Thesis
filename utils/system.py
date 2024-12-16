import torch
from abc import ABC, abstractmethod
from scipy.integrate import odeint
from Utils import sample_with_noise
import plotly.graph_objects as go

class System:
    def __init__(self, in_dim, out_dim, **params):
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Set parameters
        for key, value in params.items():
            setattr(self, key, value)

        # self.set_boundary_condition_points()
        # self.get_data_points()
        # self.get_collocation_points()

    # @property
    # def z_b(self):
    #     return self._z_b
    
    # @property
    # def U_b(self):
    #     return self._U_b
    
    # @property
    # def z_d(self):
    #     return self._z_d
    
    # @property
    # def U_d(self):
    #     return self._U_d
    
    # @property
    # def z_c(self):
    #     return self._z_c

    # def set_boundary_condition_points(self):
    #     self._z_b = torch.empty(0, self.in_dim)
    #     self._U_b = torch.empty(0, self.out_dim)

    # def get_data_points(self):
    #     self._z_d = torch.empty(0, self.in_dim)
    #     self._U_d = torch.empty(0, self.out_dim)

    # def get_collocation_points(self):
    #     self._z_c = torch.empty(0, self.in_dim)


    @abstractmethod
    def pde_residual(self, z, U, **params):
        """
        This method must be implemented.
        """
        pass

    @abstractmethod
    def evaluate_boundary(self, z, U, **params):
        """
        This method must be implemented.
        """
        pass

    def to(self, device):
    # Move all attributes to device
        for attr in self.__dict__.keys():
            if torch.is_tensor(getattr(self, attr)):
                setattr(self, attr, getattr(self, attr).to(device))



class LotkaVolterra(System):
    def __init__(self, alpha, beta, gamma, delta, X0, time_int=[0, 25], N=100, train_test=0.8):
        super().__init__(in_dim=1, out_dim=2,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            X0=X0
        )

        self.generate_boundary_points()
        self.generate_data_points(time_int, N, train_test)
        self.generate_collocation_points()


    def generate_boundary_points(self):

        self.z_b = torch.tensor([[0.0]])
        self.U_b = self.X0.unsqueeze(0)


    def generate_data_points(self, time_int, N, train_test):

        def f(X, t, alpha, beta, gamma, delta):
            x, y = X
            dxdt = alpha*x - beta*x*y
            dydt = -delta*y + gamma*x*y
            return [dxdt, dydt]
        
        self.t = torch.linspace(time_int[0], time_int[1], N)
        self.X = torch.tensor(odeint(f, self.X0, self.t, (self.alpha, self.beta, self.gamma, self.delta)), dtype=torch.float32)
        self.train_idx = torch.arange(0, train_test*N, dtype=torch.long)
        self.test_idx = torch.arange(train_test*N, N, dtype=torch.long)

        # Sample subset and add noise
        t_d, X_d = sample_with_noise(10, self.t[self.train_idx], self.X, epsilon=5e-3)

        self.z_d = t_d.unsqueeze(-1)
        self.U_d = X_d


    def generate_collocation_points(self):
        self.z_c = self.t[self.train_idx].unsqueeze(-1).requires_grad_(True)
        
    def extend_collocation_points(self):
        self.z_c = self.t.unsqueeze(-1).requires_grad_(True)

    def evaluate_boundary(self, u: torch.nn.Module):
        return u(self.z_b), self.U_b
    
    def evaluate_data(self, u: torch.nn.Module):
        return u(self.z_d), self.U_d
               
    def pde_residual(self, u: torch.nn.Module, params):

        alpha = params['alpha'] if 'alpha' in params else self.alpha
        beta = params['beta'] if 'beta' in params else self.beta
        gamma = params['gamma'] if 'gamma' in params else self.gamma
        delta = params['delta'] if 'delta' in params else self.delta

        U = u(self.z_c)
        dUdt = torch.cat([
        torch.autograd.grad(outputs=U[:, i], inputs=self.z_c, grad_outputs=torch.ones_like(U[:, i]), create_graph=True)[0]
        for i in range(U.shape[1])
        ], dim=-1)

        return torch.stack([
            dUdt[:, 0] - alpha*U[:, 0] + beta*U[:, 0]*U[:, 1],
            dUdt[:, 1] + delta*U[:, 1] # - gamma*U[:, 0]*U[:, 1] <-- Estimate this
        ], dim=-1)

    def plot_solution(self, u, G):
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

    
    def __str__(self):
        # Print pde
        pde = f"dx/dt = {self.alpha}*x - {self.beta}*x*y\n"
        pde += f"dy/dt = {self.gamma}*x*y - {self.delta}*y"
        return pde