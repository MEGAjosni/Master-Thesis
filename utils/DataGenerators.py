from scipy.integrate import odeint
import torch

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