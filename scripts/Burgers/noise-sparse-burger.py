import torch
import sys
from scipy.stats import qmc # For hypercube sampling
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Set the seed for reproducibility
torch.manual_seed(42)

# Add the parent directory of the script (i.e., project/) to sys.path
sys.path.append('utils')
from UPINN import UPINN
from Architectures import FNN
from Utils import RAD_sampler, SINDy_sklearn, SoftAdapt
from BurgerData import BurgerData
from DataGenerators import sample_collocation_points



# Initial condition
N_ic = 100
u0 = lambda x: -torch.sin(torch.pi * x)
x0 = torch.linspace(-1, 1, N_ic)
t0 = torch.zeros_like(x0)
X0 = torch.stack((t0, x0), dim=-1)
U0 = u0(x0).reshape(-1, 1)

# Boundary condition
N_bc = 100
uL = lambda t: torch.zeros_like(t)
uR = lambda t: torch.zeros_like(t)
tL = torch.linspace(0, 1, N_bc)
tR = torch.linspace(0, 1, N_bc)
xL = -torch.ones_like(tL)
xR = torch.ones_like(tR)
XL = torch.stack((tL, xL), dim=-1)
XR = torch.stack((tR, xR), dim=-1)

# All boundary conditions
Xbc = torch.cat((X0, XL, XR), dim=0)
Ubc = torch.cat((U0, uL(tL).reshape(-1, 1), uR(tR).reshape(-1, 1)), dim=0)

# Collocation points
N_coll = 10000
Xc = sample_collocation_points(N_coll, 2, lb=[0, -1], ub=[1, 1])

for sparse_level in [0.01, 0.05, 0.1, 0.5, 1.0]:
    for noise_level in [0.0, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0]:

        N = 100
        m = int(N*sparse_level)
        s = int(N/(m+1))
        time_slices = np.linspace(s, N-s-1, int(N*sparse_level), dtype=int)

        # Load data
        # noise_level = 0.0
        # time_slices = [50]
        data = BurgerData(time_slices=time_slices, noise_level=noise_level)
        Xd, Ud = data.data_points

        class SparseBurger(torch.nn.Module):
            def __init__(self, nu):
                super(SparseBurger, self).__init__()
                self.nu = nu
            
            def forward(self, X, u):

                d2udxt = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
                u_t = d2udxt[:,0:1]
                u_x = d2udxt[:,1:2]

                u_xx = torch.autograd.grad(d2udxt, X, torch.ones_like(d2udxt), create_graph=True)[0][:,1:2]

                return u_t - self.nu*u_xx


        def F_in(X, U):
            dudxt = torch.autograd.grad(U, X, torch.ones_like(U), create_graph=True)[0]
            u_t = dudxt[:,0:1]
            u_x = dudxt[:,1:2]
            return torch.cat((U, u_t, u_x), dim=-1)

        class UPINN(UPINN):

            def refine_collocation_points(self):
                N = 50*N_coll
                D = N_coll
                k = 0.5
                c = 0.1

                Xc = sample_collocation_points(N, 2, lb=[0, -1], ub=[1, 1])
                Xc.requires_grad_(True)

                # Compute the residual
                u = self.u(Xc)
                F_input = F_in(Xc, u)
                residual = abs(self.N(Xc, u) + self.F(F_input))
                
                self.collocation_points = RAD_sampler(Xc, residual, D, k, c)
            
            def score(self):
                return torch.nn.MSELoss()(self.predict(data.data_points_full[0]), data.data_points_full[1]).item()
            
            def score_residual(self):
                z = torch.clone(data.data_points_full[0]).requires_grad_(True)
                u = self.u(z)
                u_x = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0][:, 1:2]
                return torch.nn.MSELoss()(self.F(F_in(z, u)), -u_x*u).item()


            def F_input(self, X, U):
                dudxt = torch.autograd.grad(U, X, torch.ones_like(U), create_graph=True)[0]
                u_t = dudxt[:,0:1]
                u_x = dudxt[:,1:2]
                return torch.cat((U, u_t, u_x), dim=-1)
            
            def get_loss(self):
                bc_loss = self.bc_loss()
                data_loss = self.data_loss()
                pde_loss = self.pde_loss()
                loss = bc_loss + data_loss + pde_loss
                return loss, bc_loss, data_loss, pde_loss
            

        hidden = [20] * 8

        u = FNN(
            dims=[2, *hidden, 1],
            hidden_act=torch.nn.Tanh(),
            output_act=torch.nn.Identity(),
        )

        F = FNN(
            dims=[3, *hidden, 1],
            hidden_act=torch.nn.Tanh(),
            output_act=torch.nn.Identity(),
        )

        nu = 0.01/torch.pi
        N = SparseBurger(nu)

        # Instantiate the UPINN
        upinn = UPINN(u, N, F, boundary_points=(Xbc, Ubc), data_points=data.data_points, collocation_points=Xc)
        # upinn = UPINN(u, N, F, boundary_points=(Xbc, Ubc), collocation_points=Xc)

        adamw = torch.optim.AdamW(upinn.parameters(), lr=1e-3)
        lbfgs = torch.optim.LBFGS(upinn.parameters(), history_size=20, max_iter=10)

        upinn.optimizer = adamw
        upinn.train_loop(20000)

        name = 'burger-sparse'+str(sparse_level)+'-noise'+str(noise_level)+'AdamW'

        upinn.save(name=name, path='models/')

        # Write the scores to a file
        with open('models/scores.txt', 'a') as f:
            f.write(name + ' ' + str(upinn.score()) + ' ' + str(upinn.score_residual()) + '\n')