import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pysindy as ps
from kan import KAN

# Set the seed for reproducibility
torch.manual_seed(42)

# Add the parent directory of the script (i.e., project/) to sys.path
sys.path.append('utils')
from upinn import UPINN
from architectures import FNN, ResNet
from LotkaVolterraData import LotkaVolterraData
from utils import SINDy_sklearn, sample_collocation_points

# This problem is so small that the GPU overhead is slower than using the CPU
device = 'cpu'


for Ndata in [5, 10, 20, 40]:
    for noise_level, noise_level_str in zip([0.0, 0.001, 0.01, 0.1], ['0000', '0001', '0010', '0100']):

        # Data
        alpha, beta, gamma, delta = 2/3, 4/3, 1.0, 1.0
        X0 = torch.tensor([1.0, 1.0]).reshape(-1, 2)

        Nd = Ndata # Number of data points

        time_int = [0, 10]
        spacing = (time_int[1] - time_int[0]) / Nd

        data_time_points = np.linspace(time_int[0]+spacing, time_int[1], Nd)

        data = LotkaVolterraData(X0, alpha, beta, gamma, delta, time_int, time_points=data_time_points, noise_level=noise_level)

        td = data.td
        Xd = data.Xd


        # Boundary conditions
        tbc = torch.tensor([[0.0]])
        Xbc = X0


        # Collocation points
        N_coll = 1000
        tc = sample_collocation_points(N_coll, 1, lb=[time_int[0]], ub=[time_int[1]], method='grid')


        class SparseLV(torch.nn.Module):
            
            def __init__(self, params):
                super(SparseLV, self).__init__()
                self.alpha = params['alpha']
                self.beta = params['beta']
                self.delta = params['delta']
                self.gamma = params['gamma']
                self.log = dict()

            def forward(self, z, U):

                dUdt = torch.cat([
                torch.autograd.grad(outputs=U[:, i], inputs=z, grad_outputs=torch.ones_like(U[:, i]), create_graph=True)[0]
                for i in range(U.shape[1])
                ], dim=-1)

                # self.log.setdefault('alpha', []).append(self.alpha.item())
                # self.log.setdefault('delta', []).append(self.delta.item())

                return torch.stack([
                    (dUdt[:, 0] - self.alpha*U[:, 0]), # + beta*U[:, 0]*U[:, 1],
                    (dUdt[:, 1] + self.delta*U[:, 1]) # - gamma*U[:, 0]*U[:, 1] # <-- Estimate this
                ], dim=-1)


        params = dict(
            alpha=alpha,
            # alpha=torch.nn.Parameter(torch.tensor(1.0)),
            beta=beta,
            # delta=torch.nn.Parameter(torch.tensor(1.0)),
            delta=delta,
            gamma=gamma
        )

        N = SparseLV(params)


        class LV_UPINN(UPINN):

            # def set_psi(self):
            #     self.psi = torch.nn.Parameter(torch.tensor(1.0))

            # def get_loss(self):
            #     bc_loss = self.bc_loss()
            #     data_loss = self.data_loss()
            #     pde_loss = self.pde_loss()
            #     lambdas = SoftAdapt(**self.softadapt_kwargs)(torch.tensor([bc_loss, data_loss, pde_loss]))
            #     loss = lambdas[0]*bc_loss + lambdas[1]*data_loss + lambdas[2]*pde_loss

            #     return loss, bc_loss, data_loss, pde_loss

            def F_input(self, z, U):
                return U

            def score(self):
                u_pred = self.u(data.t_full)

                L2_rel_error = torch.sqrt(torch.mean((u_pred - data.X_full)**2) / torch.mean(data.X_full**2))
                return L2_rel_error.item()

                # return torch.nn.MSELoss()(u_pred, data.X_full)
            
            def score_residual(self):
                u_pred = self.u(data.t_full)
                F_pred = self.F(self.F_input(data.t_full, u_pred))

                # self.shared_res = True
                # if self.shared_res:
                #     F_pred = torch.cat((F_pred, self.psi*F_pred), dim=1)
                    
                F_exp_1 = beta*u_pred[:, 0]*u_pred[:, 1]
                F_exp_2 = -gamma*u_pred[:, 0]*u_pred[:, 1]
                F_exp = torch.stack([F_exp_1, F_exp_2], dim=-1)

                L2_rel_error = torch.sqrt(torch.mean((F_pred - F_exp)**2) / torch.mean(F_exp**2))
                return L2_rel_error.item()

            
            def pde_loss(self):
                # self.log.setdefault('psi', []).append(self.psi.item())
                if self.collocation_points is not None:
                    U_c = self.u(self.collocation_points)
                    res = self.F(self.F_input(self.collocation_points, U_c))

                    # self.shared_res = True
                    # if self.shared_res:
                    #     res = torch.cat((res, self.psi*res), dim=1)

                    known = self.N(self.collocation_points, U_c)
                    pde_loss = torch.nn.MSELoss()(known, -res) if self.collocation_points.shape[0] > 0 else torch.tensor(0.0)
                else: pde_loss = torch.tensor(0.0)
                return pde_loss


        # Define model architectures
        hidden = [16] * 4
        u = FNN(
            dims=[1, *hidden, 2],
            hidden_act=torch.nn.Tanh(),
            output_act=torch.nn.SiLU(),
        )
        F = FNN(
            dims=[2, *hidden, 2],
            hidden_act=torch.nn.Tanh(),
            output_act=torch.nn.Identity(),
        )


        model = LV_UPINN(u, N, F, boundary_points=(tbc, Xbc), data_points=(td, Xd), collocation_points=tc)


        adamw = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adamw, factor=0.5, patience=200, threshold=1e-3)
        model.scheduler = scheduler
        model.optimizer = adamw


        model.train_loop(100000, device='cpu')

        name = 'lv_baseline_data'+str(Ndata)+'_noise'+noise_level_str
        model.save(name, 'models')
        np.save('loss/'+name+'_loss.npy', model.log['loss'])

        scores = np.array([model.score(), model.score_residual()])

        np.save('loss/'+name+'_score.npy', scores)
        np.save('loss/'+name+'_log.npy', model.log)