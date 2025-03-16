import numpy as np
import torch
from scipy.integrate import odeint

class SIR:
    def __init__(self, betas, gamma, X0):
        self.alpha, self.betas, self.gamma = betas, gamma
        self.X0 = X0

    def f(self, X, t, betas, gamma):

        S, I1, I2, R = X
        beta1, beta2 = betas        

        dSdt = - beta1*S*I1 - beta2*S*I2
        dI1dt = beta1*S*I1 - gamma*I1
        dI2dt = beta2*S*I2 - gamma*I2
        dRdt = gamma*(I1 + I2)

        return [dSdt, dI1dt, dI2dt, dRdt]
    
    def solve(self, t):
        out = odeint(self.f, self.X0, t.squeeze(), (self.alpha, self.beta, self.gamma, self.delta))
        return torch.tensor(out, dtype=torch.float32)


# class SIRData:
#     def __init__(self, X0, beta1, beta2, gamma1, gamma2, time_int, N=1000, time_points=None, noise_level=None, shuffle=False):
        

#         ############################################
#         ###     Generate data from SIR model     ###
#         ############################################
#         ### dS/dt = -beta*S*I                    ###
#         ### dI/dt = beta*S*I - gamma*I - theta*I ###
#         ### dR1/dt = gamma*I                     ###
#         ### dR2/dt = theta*I                     ###
#         ############################################
#         # alpha, beta, gamma, delta = 1.3, 0.9, 0.8, 1.8
#         # x0, y0 = 0.44249296, 4.6280594

#         self.t_full = torch.linspace(time_int[0], time_int[1], N).reshape(-1, 1)

#         def f(X, t, beta1, beta2, gamma1, gamma2):

#             S, I1, I2, R = X

#             dSdt = - beta1*S*I1 - beta2*S*I2
#             dI1dt = beta1*S*I1 - gamma1*I1
#             dI2dt = beta2*S*I2 - gamma2*I2
#             dRdt = gamma1*I1 + gamma2*I2

#             return [dSdt, dI1dt, dI2dt, dRdt]

#         self.X_full = torch.tensor(odeint(f, X0.squeeze(), self.t_full.squeeze(), (beta1, beta2, gamma1, gamma2)), dtype=torch.float32).reshape(-1, len(X0))
        
#         # Add 0.0 to start of time_points
#         time_points = [0.0] + list(time_points) if time_points is not None else None
#         self.td = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1)[1:] if time_points is not None else None
#         self.Xd = torch.tensor(odeint(f, X0.squeeze(), time_points, (beta1, beta2, gamma1, gamma2)), dtype=torch.float32).reshape(-1, len(X0))[1:] if time_points is not None else None
       
#         # Add noise
#         if noise_level:
#             self.Xd += noise_level * torch.mean(abs(self.Xd)) * torch.randn(*self.Xd.shape)
        
#         # Shuffle data
#         if shuffle:
#             idx = torch.randperm(self.Xd.shape[0])
#             self.Xd, self.td = self.Xd[idx], self.td[idx]

#     def __len__(self):
#         return len(self.Xd)
    
#     def __getitem__(self, idx):
#         return self.td[idx], self.Xd[idx]
    


class SIRData:
    def __init__(self, X0, beta, gamma, theta, time_int, N=1000, time_points=None, noise_level=None, shuffle=False):
        

        ############################################
        ###     Generate data from SIR model     ###
        ############################################
        ### dS/dt = -beta*S*I                    ###
        ### dI/dt = beta*S*I - gamma*I - theta*I ###
        ### dR1/dt = gamma*I                     ###
        ### dR2/dt = theta*I                     ###
        ############################################
        # alpha, beta, gamma, delta = 1.3, 0.9, 0.8, 1.8
        # x0, y0 = 0.44249296, 4.6280594

        self.t_full = torch.linspace(time_int[0], time_int[1], N).reshape(-1, 1)

        def f(X, t, beta, gamma, theta):

            S, I, R1, R2 = X

            dSdt = - beta*S*I
            dIdt = beta*S*I - gamma*I - theta*I
            dR1dt = gamma*I
            dR2dt = theta*I

            return [dSdt, dIdt, dR1dt, dR2dt]

        self.X_full = torch.tensor(odeint(f, X0.squeeze(), self.t_full.squeeze(), (beta, gamma, theta)), dtype=torch.float32).reshape(-1, len(X0))
        
        # Add 0.0 to start of time_points
        time_points = [0.0] + list(time_points) if time_points is not None else None
        self.td = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1)[1:] if time_points is not None else None
        self.Xd = torch.tensor(odeint(f, X0.squeeze(), time_points, (beta, gamma, theta)), dtype=torch.float32).reshape(-1, len(X0))[1:] if time_points is not None else None
       
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


class CyclicSIRData:
    def __init__(self, X0, beta, gamma, theta, time_int, N=1000, time_points=None, noise_level=None, shuffle=False):
        

        ############################################
        ###     Generate data from SIR model     ###
        ############################################
        ### dS/dt = -beta*S*I                    ###
        ### dI/dt = beta*S*I - gamma*I - theta*I ###
        ### dR1/dt = gamma*I                     ###
        ### dR2/dt = theta*I                     ###
        ############################################
        # alpha, beta, gamma, delta = 1.3, 0.9, 0.8, 1.8
        # x0, y0 = 0.44249296, 4.6280594

        self.t_full = torch.linspace(time_int[0], time_int[1], N).reshape(-1, 1)

        def f(X, t, beta, gamma, theta):

            S, I, R = X

            dSdt = - beta*S*I + theta*R
            dIdt = beta*S*I - gamma*I
            dRdt = gamma*I - theta*R

            return [dSdt, dIdt, dRdt]

        self.X_full = torch.tensor(odeint(f, X0.squeeze(), self.t_full.squeeze(), (beta, gamma, theta)), dtype=torch.float32).reshape(-1, len(X0))
        
        # Add 0.0 to start of time_points
        time_points = [0.0] + list(time_points) if time_points is not None else None
        self.td = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1)[1:] if time_points is not None else None
        self.Xd = torch.tensor(odeint(f, X0.squeeze(), time_points, (beta, gamma, theta)), dtype=torch.float32).reshape(-1, len(X0))[1:] if time_points is not None else None
       
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