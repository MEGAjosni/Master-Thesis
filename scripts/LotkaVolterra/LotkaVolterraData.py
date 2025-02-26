import numpy as np
import torch
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
        out = odeint(self.f, self.X0, t.squeeze(), (self.alpha, self.beta, self.gamma, self.delta))
        return torch.tensor(out, dtype=torch.float32)


class LotkaVolterraData:
    def __init__(self, X0, alpha, beta, gamma, delta, time_int, N=1000, time_points=None, noise_level=None, shuffle=False):
        

        ###############################################
        ### Generate data from Lotka-Volterra model ###
        ###############################################
        ###   dx/dt = alpha*x - beta*x*y            ###
        ###   dy/dt = gamma*x*y - delta*y           ###
        ###############################################
        # alpha, beta, gamma, delta = 1.3, 0.9, 0.8, 1.8
        # x0, y0 = 0.44249296, 4.6280594

        self.t_full = torch.linspace(time_int[0], time_int[1], N).reshape(-1, 1)

        def f(X, t, alpha, beta, gamma, delta):
            x, y = X
            dxdt = alpha*x - beta*x*y
            dydt = -delta*y + gamma*x*y
            return [dxdt, dydt]

        self.X_full = torch.tensor(odeint(f, X0.squeeze(), self.t_full.squeeze(), (alpha, beta, gamma, delta)), dtype=torch.float32).reshape(-1, 2)
        
        # Add 0.0 to start of time_points
        time_points = [0.0] + list(time_points) if time_points is not None else None
        self.td = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1)[1:] if time_points is not None else None
        self.Xd = torch.tensor(odeint(f, X0.squeeze(), time_points, (alpha, beta, gamma, delta)), dtype=torch.float32).reshape(-1, 2)[1:] if time_points is not None else None

        

        # Add noise
        if noise_level:
            self.Xd += noise_level * torch.mean(abs(self.Xd)) * torch.randn(*self.Xd.shape)
        
        # Shuffle data
        if shuffle:
            idx = torch.randperm(self.Xd.shape[0])
            self.Xd, self.td = self.Xd[idx], self.td[idx]

    def __len__(self):
        return len(self.Xd)
    
    def __getitem__(self, idx):
        return self.td[idx], self.Xd[idx]