from scipy.integrate import odeint
import torch
import numpy as np

def sample_with_noise(N, t, X, epsilon=5e-3):

    # Check if the shapes are correct and feasible amount of points are requested
    assert len(X) != len(t) or N <= len(t), "Invalid shapes or N"

    # Calculate the mean of the data
    X_bar = torch.mean(X, dim=0)

    # Sample N evenly spaced points from the data
    idx = torch.linspace(len(t)//N, len(t)-1, N, dtype=torch.int)
    t, X = t[idx], X[idx]

    # Add noise to the data
    X_noise = X + epsilon * X_bar * torch.randn(*X.shape)

    return t, X_noise


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
    

class SIR:
    def __init__(self, beta, gamma, X0):
        self.beta, self.gamma = beta, gamma
        self.X0 = X0

    def f(self, X, t, beta, gamma):
        S = X[0]
        I = X[1:-1]
        R = X[-1]

        dSdt = np.sum(-np.array(beta) * S * I)
        dIdt = np.array(beta) * S * I - np.array(gamma) * I
        dRdt = np.sum(np.array(gamma) * I)
        
        return [dSdt] + list(dIdt) + [dRdt]
    
    def solve(self, t):
        out = odeint(self.f, self.X0, t.squeeze(), (self.beta, self.gamma))
        return torch.tensor(out, dtype=torch.float32)