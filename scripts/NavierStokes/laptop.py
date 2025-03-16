# # Relevant imports
# import torch
# import sys
# # import matplotlib.pyplot as plt
# import numpy as np
# # from kan import KAN

# # Set the seed for reproducibility
# torch.manual_seed(42)

# # Set cwd to this script's directory
# import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # Add the parent directory of the script (i.e., project/) to sys.path
# sys.path.append('../../utils')
# from upinn import UPINN
# from architectures import FNN, StackedNN, ResNet, count_parameters
# from utils import RAD_sampler, sample_collocation_points
# from NavierStokesData import NavierStokesData





# for sample_size in [100, 500, 1000]:
#     name = 'ns-upinn-with-pressuredata-FNN64-sa02-RAD-alt-'+str(sample_size)+'-noise001_new'

#     print(f"Training {name}")

#     # Data points
#     data = NavierStokesData(samplesize=sample_size, noise_level=0.01)
#     Zd = data.Zd
#     Ud = data.Ud

#     # Collocation points
#     N_coll = 10000
#     Xc = sample_collocation_points(N_coll, 3, [1, -2, 0], [8, 2, 20], method='sobol')

#     def compute_grad(outputs, inputs):
#         return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]

#     class NavierStokes(torch.nn.Module):
#         def __init__(self, lambda1, lambda2):
#             super(NavierStokes, self).__init__()
#             self.lambda1 = lambda1
#             self.lambda2 = lambda2


#         def forward(self, Z, U):

#             psi = U[:, 0:1]
#             p = U[:, 1:2]

#             U_z = compute_grad(psi, Z)

#             psi_x = U_z[:, 0:1]
#             psi_y = U_z[:, 1:2]

#             u = psi_y
#             v = -psi_x

#             u_z = compute_grad(u, Z)
#             u_x = u_z[:, 0:1]
#             u_y = u_z[:, 1:2]
#             u_t = u_z[:, 2:3]

#             v_z = compute_grad(v, Z)
#             v_x = v_z[:, 0:1]
#             v_y = v_z[:, 1:2]
#             v_t = v_z[:, 2:3]

#             p_z = compute_grad(p, Z)
#             p_x = p_z[:, 0:1]
#             p_y = p_z[:, 1:2]

#             u_xx = compute_grad(u_x, Z)[:, 0:1]
#             u_yy = compute_grad(u_y, Z)[:, 1:2]

#             v_xx = compute_grad(v_x, Z)[:, 0:1]
#             v_yy = compute_grad(v_y, Z)[:, 1:2]

#             f = u_t + self.lambda1 * (u * u_x + v * u_y) + p_x # - self.lambda2 * (u_xx + u_yy)
#             g = v_t + self.lambda1 * (u * v_x + v * v_y) + p_y # - self.lambda2 * (v_xx + v_yy)

#             return torch.cat([f, g], dim=1)

#     lambda1 = 1.0
#     lambda2 = 0.01

#     N = NavierStokes(lambda1, lambda2)


#     class NavierStokesUPINN(UPINN):
        
#         def score(self):

#             Xtest = data.Zd_full.requires_grad_(True)

#             U_pred = self.u(Xtest)
#             psi_z = compute_grad(U_pred[:, 0], Xtest)
#             psi_y = psi_z[:, 1:2]
#             psi_x = psi_z[:, 0:1]

#             data_pred = torch.cat([psi_y, -psi_x], dim=1)
#             data_true = torch.mean((data_pred - data.Ud_full[:, 0:2])**2)

#             return torch.nn.MSELoss()(data_pred, data_true)
        

#         def score_residual(self):
            
#             x_true = np.linspace(1, 8, 100)
#             y_true = np.linspace(-2, 2, 50)
#             u_true = data.Ud_full[:, 0].reshape(50,100)
#             v_true = data.Ud_full[:, 1].reshape(50,100)

#             # True residual
#             u_x = np.gradient(u_true, x_true, axis=1)
#             u_y = np.gradient(u_true, y_true, axis=0)

#             v_x = np.gradient(v_true, x_true, axis=1)
#             v_y = np.gradient(v_true, y_true, axis=0)

#             u_xx = np.gradient(u_x, x_true, axis=1)
#             u_yy = np.gradient(u_y, y_true, axis=0)

#             v_xx = np.gradient(v_x, x_true, axis=1)
#             v_yy = np.gradient(v_y, y_true, axis=0)

#             res_true_f = - lambda2 * (u_xx + u_yy)
#             res_true_g = - lambda2 * (v_xx + v_yy)

#             res_true = np.concatenate([res_true_f.reshape(-1,1), res_true_g.reshape(-1,1)], axis=1)

#             z_test = torch.tensor(np.array(np.meshgrid(x_true, y_true)).T.reshape(-1,2)).float()
#             u_test = self.u(z_test)
#             res_pred = self.N(z_test, u_test).detach().numpy()
#             res_pred = res_pred.reshape(50,100,2)

#             residual_loss = np.mean((res_true - res_pred)**2)

#             return residual_loss


#         def data_loss(self):

#             if self.data_points is not None:
                
#                 self.data_points[0].requires_grad_(True)

#                 Ud = self.u(self.data_points[0])

#                 psi_z = compute_grad(Ud[:, 0], self.data_points[0])
#                 psi_y = psi_z[:, 1:2]
#                 psi_x = psi_z[:, 0:1]

#                 # data_pred = torch.cat([psi_y, -psi_x], dim=1)
#                 # data_loss = torch.mean((data_pred - self.data_points[1][:, 0:2])**2)

#                 data_pred = torch.cat([psi_y, -psi_x, Ud[:, 1:2]], dim=1)
#                 data_loss = torch.mean((data_pred - self.data_points[1][:, 0:3])**2)

#                 # self.log.setdefault("lambda1", []).append(lambda1.item())
#                 # self.log.setdefault("lambda2", []).append(lambda2.item())

#             else: data_loss = torch.tensor(0.0)

#             return data_loss
        
#         def F_input(self, Z, U):
            
#             psi = U[:, 0:1]
            
#             U_z = compute_grad(psi, Z)

#             psi_x = U_z[:, 0:1]
#             psi_y = U_z[:, 1:2]

#             u = psi_y
#             v = -psi_x

#             return torch.cat([Z, u, v], dim=1)


#         def refine_collocation_points(self):
#             N = 10*N_coll
#             D = N_coll
#             k = 0.5
#             c = 0.1

#             Xc = sample_collocation_points(N, 3, lb=[1, -2, 0], ub=[8, 2, 20], method='sobol').requires_grad_(True)

#             # Compute the residual
#             U = self.u(Xc)
#             residual = torch.sum(torch.abs(self.F(self.F_input(Xc, U)) + self.N(Xc, U)), dim=1)

#             self.collocation_points = RAD_sampler(Xc, residual, D, k, c) # RAD
        


#     hidden_u = [64] * 8

#     u = FNN(
#         dims=[3, *hidden_u, 2],
#         hidden_act=torch.nn.Tanh(),
#         output_act=torch.nn.Identity(),
#     )

#     hidden_F = [64] * 8

#     F = FNN(
#         dims=[5, *hidden_F, 2],
#         hidden_act=torch.nn.Tanh(),
#         output_act=torch.nn.Identity(),
#     )


#     model = NavierStokesUPINN(u, N, F, data_points=(Zd, Ud), collocation_points=Xc)
#     model.softadapt_kwargs = dict(beta=0.2, loss_weigthed=False)

#     adam = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1000, gamma=0.8)
#     model.scheduler = scheduler

#     lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1)


#     for i in range(15):
#         model.optimizer = adam
#         model.train_loop(1000)
#         model.optimizer = lbfgs
#         model.train_loop(50)
#         model.refine_collocation_points()
#         if not torch.isnan(model.get_loss()[0]).item():
#             model.save(name, 'models', overwrite=True)
#             np.save('loss/loss_'+name, model.log['loss'])

#     if not torch.isnan(model.get_loss()[0]).item():
#         model.optimizer = adam
#         model.train_loop(20000)
#         model.save(name, 'models', overwrite=True)
#         np.save('loss/loss_'+name, model.log['loss'])

#     model.optimizer = lbfgs
#     model.train_loop(100)
#     if not torch.isnan(model.get_loss()[0]).item():
#         model.save(name, 'models', overwrite=True)
#         np.save('loss/loss_'+name, model.log['loss'])




# # Relevant imports
# import torch
# import sys
# # import matplotlib.pyplot as plt
# import numpy as np
# # from kan import KAN

# # Set the seed for reproducibility
# torch.manual_seed(42)

# # Set cwd to this script's directory
# import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # Add the parent directory of the script (i.e., project/) to sys.path
# sys.path.append('../../utils')
# from upinn import UPINN
# from architectures import FNN, StackedNN, ResNet, count_parameters
# from utils import RAD_sampler, sample_collocation_points
# from NavierStokesData import NavierStokesData





# for sample_size in [100, 500, 1000, 2000]:
#     name = 'ns-reverse-upinn-with-pressuredata-FNN64-sa02-RAD-alt-'+str(sample_size)+'-noise001_new'

#     print(f"Training {name}")

#     # Data points
#     data = NavierStokesData(samplesize=sample_size, noise_level=0.01)
#     Zd = data.Zd
#     Ud = data.Ud

#     # Collocation points
#     N_coll = 10000
#     Xc = sample_collocation_points(N_coll, 3, [1, -2, 0], [8, 2, 20], method='sobol')

#     def compute_grad(outputs, inputs):
#         return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]

#     class NavierStokes(torch.nn.Module):
#         def __init__(self, lambda1, lambda2):
#             super(NavierStokes, self).__init__()
#             self.lambda1 = lambda1
#             self.lambda2 = lambda2


#         def forward(self, Z, U):

#             psi = U[:, 0:1]
#             p = U[:, 1:2]

#             U_z = compute_grad(psi, Z)

#             psi_x = U_z[:, 0:1]
#             psi_y = U_z[:, 1:2]

#             u = psi_y
#             v = -psi_x

#             u_z = compute_grad(u, Z)
#             u_x = u_z[:, 0:1]
#             u_y = u_z[:, 1:2]
#             u_t = u_z[:, 2:3]

#             v_z = compute_grad(v, Z)
#             v_x = v_z[:, 0:1]
#             v_y = v_z[:, 1:2]
#             v_t = v_z[:, 2:3]

#             p_z = compute_grad(p, Z)
#             p_x = p_z[:, 0:1]
#             p_y = p_z[:, 1:2]

#             u_xx = compute_grad(u_x, Z)[:, 0:1]
#             u_yy = compute_grad(u_y, Z)[:, 1:2]

#             v_xx = compute_grad(v_x, Z)[:, 0:1]
#             v_yy = compute_grad(v_y, Z)[:, 1:2]

#             f = u_t + p_x - self.lambda2 * (u_xx + u_yy)
#             g = v_t + p_y - self.lambda2 * (v_xx + v_yy)

#             return torch.cat([f, g], dim=1)

#     lambda1 = 1.0
#     lambda2 = 0.01

#     N = NavierStokes(lambda1, lambda2)


#     class NavierStokesUPINN(UPINN):
        
#         def score(self):

#             Xtest = data.Zd_full.requires_grad_(True)

#             U_pred = self.u(Xtest)
#             psi_z = compute_grad(U_pred[:, 0], Xtest)
#             psi_y = psi_z[:, 1:2]
#             psi_x = psi_z[:, 0:1]

#             data_pred = torch.cat([psi_y, -psi_x], dim=1)
#             data_true = torch.mean((data_pred - data.Ud_full[:, 0:2])**2)

#             return torch.nn.MSELoss()(data_pred, data_true)
        

#         def score_residual(self):
            
#             x_true = np.linspace(1, 8, 100)
#             y_true = np.linspace(-2, 2, 50)
#             u_true = data.Ud_full[:, 0].reshape(50,100)
#             v_true = data.Ud_full[:, 1].reshape(50,100)

#             # True residual
#             u_x = np.gradient(u_true, x_true, axis=1)
#             u_y = np.gradient(u_true, y_true, axis=0)

#             v_x = np.gradient(v_true, x_true, axis=1)
#             v_y = np.gradient(v_true, y_true, axis=0)

#             u_xx = np.gradient(u_x, x_true, axis=1)
#             u_yy = np.gradient(u_y, y_true, axis=0)

#             v_xx = np.gradient(v_x, x_true, axis=1)
#             v_yy = np.gradient(v_y, y_true, axis=0)

#             res_true_f = - lambda2 * (u_xx + u_yy)
#             res_true_g = - lambda2 * (v_xx + v_yy)

#             res_true = np.concatenate([res_true_f.reshape(-1,1), res_true_g.reshape(-1,1)], axis=1)

#             z_test = torch.tensor(np.array(np.meshgrid(x_true, y_true)).T.reshape(-1,2)).float()
#             u_test = self.u(z_test)
#             res_pred = self.N(z_test, u_test).detach().numpy()
#             res_pred = res_pred.reshape(50,100,2)

#             residual_loss = np.mean((res_true - res_pred)**2)

#             return residual_loss


#         def data_loss(self):

#             if self.data_points is not None:
                
#                 self.data_points[0].requires_grad_(True)

#                 Ud = self.u(self.data_points[0])

#                 psi_z = compute_grad(Ud[:, 0], self.data_points[0])
#                 psi_y = psi_z[:, 1:2]
#                 psi_x = psi_z[:, 0:1]

#                 # data_pred = torch.cat([psi_y, -psi_x], dim=1)
#                 # data_loss = torch.mean((data_pred - self.data_points[1][:, 0:2])**2)

#                 data_pred = torch.cat([psi_y, -psi_x, Ud[:, 1:2]], dim=1)
#                 data_loss = torch.mean((data_pred - self.data_points[1][:, 0:3])**2)

#                 # self.log.setdefault("lambda1", []).append(lambda1.item())
#                 # self.log.setdefault("lambda2", []).append(lambda2.item())

#             else: data_loss = torch.tensor(0.0)

#             return data_loss
        
#         def F_input(self, Z, U):
            
#             psi = U[:, 0:1]
            
#             U_z = compute_grad(psi, Z)

#             psi_x = U_z[:, 0:1]
#             psi_y = U_z[:, 1:2]

#             u = psi_y
#             v = -psi_x

#             return torch.cat([Z, u, v], dim=1)


#         def refine_collocation_points(self):
#             N = 10*N_coll
#             D = N_coll
#             k = 0.5
#             c = 0.1

#             Xc = sample_collocation_points(N, 3, lb=[1, -2, 0], ub=[8, 2, 20], method='sobol').requires_grad_(True)

#             # Compute the residual
#             U = self.u(Xc)
#             residual = torch.sum(torch.abs(self.F(self.F_input(Xc, U)) + self.N(Xc, U)), dim=1)

#             self.collocation_points = RAD_sampler(Xc, residual, D, k, c) # RAD
        


#     hidden_u = [64] * 8

#     u = FNN(
#         dims=[3, *hidden_u, 2],
#         hidden_act=torch.nn.Tanh(),
#         output_act=torch.nn.Identity(),
#     )

#     hidden_F = [64] * 8

#     F = FNN(
#         dims=[5, *hidden_F, 2],
#         hidden_act=torch.nn.Tanh(),
#         output_act=torch.nn.Identity(),
#     )


#     model = NavierStokesUPINN(u, N, F, data_points=(Zd, Ud), collocation_points=Xc)
#     model.softadapt_kwargs = dict(beta=0.2, loss_weigthed=False)

#     adam = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1000, gamma=0.8)
#     model.scheduler = scheduler

#     lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1)


#     for i in range(15):
#         model.optimizer = adam
#         model.train_loop(1000)
#         model.optimizer = lbfgs
#         model.train_loop(50)
#         model.refine_collocation_points()
#         if not torch.isnan(model.get_loss()[0]).item():
#             model.save(name, 'models', overwrite=True)
#             np.save('loss/loss_'+name, model.log['loss'])

#     if not torch.isnan(model.get_loss()[0]).item():
#         model.optimizer = adam
#         model.train_loop(20000)
#         model.save(name, 'models', overwrite=True)
#         np.save('loss/loss_'+name, model.log['loss'])

#     model.optimizer = lbfgs
#     model.train_loop(100)
#     if not torch.isnan(model.get_loss()[0]).item():
#         model.save(name, 'models', overwrite=True)
#         np.save('loss/loss_'+name, model.log['loss'])



# Relevant imports
import torch
import sys
# import matplotlib.pyplot as plt
import numpy as np
# from kan import KAN

# Set the seed for reproducibility
torch.manual_seed(42)

# Set cwd to this script's directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory of the script (i.e., project/) to sys.path
sys.path.append('../../utils')
from upinn import UPINN
from architectures import FNN, StackedNN, ResNet, count_parameters
from utils import RAD_sampler, sample_collocation_points
from NavierStokesData import NavierStokesData





for sample_size in [500]:
    name = 'ns-reverse-upinn-with-pressuredata-FNN64-sa02-RAD-alt-'+str(sample_size)+'_new'

    print(f"Training {name}")

    # Data points
    data = NavierStokesData(samplesize=sample_size)
    Zd = data.Zd
    Ud = data.Ud

    # Collocation points
    N_coll = 10000
    Xc = sample_collocation_points(N_coll, 3, [1, -2, 0], [8, 2, 20], method='sobol')

    def compute_grad(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]

    class NavierStokes(torch.nn.Module):
        def __init__(self, lambda1, lambda2):
            super(NavierStokes, self).__init__()
            self.lambda1 = lambda1
            self.lambda2 = lambda2


        def forward(self, Z, U):

            psi = U[:, 0:1]
            p = U[:, 1:2]

            U_z = compute_grad(psi, Z)

            psi_x = U_z[:, 0:1]
            psi_y = U_z[:, 1:2]

            u = psi_y
            v = -psi_x

            u_z = compute_grad(u, Z)
            u_x = u_z[:, 0:1]
            u_y = u_z[:, 1:2]
            u_t = u_z[:, 2:3]

            v_z = compute_grad(v, Z)
            v_x = v_z[:, 0:1]
            v_y = v_z[:, 1:2]
            v_t = v_z[:, 2:3]

            p_z = compute_grad(p, Z)
            p_x = p_z[:, 0:1]
            p_y = p_z[:, 1:2]

            u_xx = compute_grad(u_x, Z)[:, 0:1]
            u_yy = compute_grad(u_y, Z)[:, 1:2]

            v_xx = compute_grad(v_x, Z)[:, 0:1]
            v_yy = compute_grad(v_y, Z)[:, 1:2]

            f = u_t + p_x - self.lambda2 * (u_xx + u_yy)
            g = v_t + p_y - self.lambda2 * (v_xx + v_yy)

            return torch.cat([f, g], dim=1)

    lambda1 = 1.0
    lambda2 = 0.01

    N = NavierStokes(lambda1, lambda2)


    class NavierStokesUPINN(UPINN):
        
        def score(self):

            Xtest = data.Zd_full.requires_grad_(True)

            U_pred = self.u(Xtest)
            psi_z = compute_grad(U_pred[:, 0], Xtest)
            psi_y = psi_z[:, 1:2]
            psi_x = psi_z[:, 0:1]

            data_pred = torch.cat([psi_y, -psi_x], dim=1)
            data_true = torch.mean((data_pred - data.Ud_full[:, 0:2])**2)

            return torch.nn.MSELoss()(data_pred, data_true)
        

        def score_residual(self):
            
            x_true = np.linspace(1, 8, 100)
            y_true = np.linspace(-2, 2, 50)
            u_true = data.Ud_full[:, 0].reshape(50,100)
            v_true = data.Ud_full[:, 1].reshape(50,100)

            # True residual
            u_x = np.gradient(u_true, x_true, axis=1)
            u_y = np.gradient(u_true, y_true, axis=0)

            v_x = np.gradient(v_true, x_true, axis=1)
            v_y = np.gradient(v_true, y_true, axis=0)

            u_xx = np.gradient(u_x, x_true, axis=1)
            u_yy = np.gradient(u_y, y_true, axis=0)

            v_xx = np.gradient(v_x, x_true, axis=1)
            v_yy = np.gradient(v_y, y_true, axis=0)

            res_true_f = - lambda2 * (u_xx + u_yy)
            res_true_g = - lambda2 * (v_xx + v_yy)

            res_true = np.concatenate([res_true_f.reshape(-1,1), res_true_g.reshape(-1,1)], axis=1)

            z_test = torch.tensor(np.array(np.meshgrid(x_true, y_true)).T.reshape(-1,2)).float()
            u_test = self.u(z_test)
            res_pred = self.N(z_test, u_test).detach().numpy()
            res_pred = res_pred.reshape(50,100,2)

            residual_loss = np.mean((res_true - res_pred)**2)

            return residual_loss


        def data_loss(self):

            if self.data_points is not None:
                
                self.data_points[0].requires_grad_(True)

                Ud = self.u(self.data_points[0])

                psi_z = compute_grad(Ud[:, 0], self.data_points[0])
                psi_y = psi_z[:, 1:2]
                psi_x = psi_z[:, 0:1]

                # data_pred = torch.cat([psi_y, -psi_x], dim=1)
                # data_loss = torch.mean((data_pred - self.data_points[1][:, 0:2])**2)

                data_pred = torch.cat([psi_y, -psi_x, Ud[:, 1:2]], dim=1)
                data_loss = torch.mean((data_pred - self.data_points[1][:, 0:3])**2)

                # self.log.setdefault("lambda1", []).append(lambda1.item())
                # self.log.setdefault("lambda2", []).append(lambda2.item())

            else: data_loss = torch.tensor(0.0)

            return data_loss
        
        def F_input(self, Z, U):
            
            psi = U[:, 0:1]
            
            U_z = compute_grad(psi, Z)

            psi_x = U_z[:, 0:1]
            psi_y = U_z[:, 1:2]

            u = psi_y
            v = -psi_x

            return torch.cat([Z, u, v], dim=1)


        def refine_collocation_points(self):
            N = 10*N_coll
            D = N_coll
            k = 0.5
            c = 0.1

            Xc = sample_collocation_points(N, 3, lb=[1, -2, 0], ub=[8, 2, 20], method='sobol').requires_grad_(True)

            # Compute the residual
            U = self.u(Xc)
            residual = torch.sum(torch.abs(self.F(self.F_input(Xc, U)) + self.N(Xc, U)), dim=1)

            self.collocation_points = RAD_sampler(Xc, residual, D, k, c) # RAD
        


    hidden_u = [64] * 8

    u = FNN(
        dims=[3, *hidden_u, 2],
        hidden_act=torch.nn.Tanh(),
        output_act=torch.nn.Identity(),
    )

    hidden_F = [64] * 8

    F = FNN(
        dims=[5, *hidden_F, 2],
        hidden_act=torch.nn.Tanh(),
        output_act=torch.nn.Identity(),
    )


    model = NavierStokesUPINN(u, N, F, data_points=(Zd, Ud), collocation_points=Xc)
    model.softadapt_kwargs = dict(beta=0.2, loss_weigthed=False)

    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1000, gamma=0.8)
    model.scheduler = scheduler

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1)


    for i in range(15):
        model.optimizer = adam
        model.train_loop(1000)
        model.optimizer = lbfgs
        model.train_loop(50)
        model.refine_collocation_points()
        if not torch.isnan(model.get_loss()[0]).item():
            model.save(name, 'models', overwrite=True)
            np.save('loss/loss_'+name, model.log['loss'])

    if not torch.isnan(model.get_loss()[0]).item():
        model.optimizer = adam
        model.train_loop(20000)
        model.save(name, 'models', overwrite=True)
        np.save('loss/loss_'+name, model.log['loss'])

    model.optimizer = lbfgs
    model.train_loop(100)
    if not torch.isnan(model.get_loss()[0]).item():
        model.save(name, 'models', overwrite=True)
        np.save('loss/loss_'+name, model.log['loss'])