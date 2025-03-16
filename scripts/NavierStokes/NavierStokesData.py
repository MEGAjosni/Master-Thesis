import scipy.io
import numpy as np
import torch

def compute_grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]

class NavierStokesData:
    def __init__(self, samplesize=5000, noise_level=0.0):

        # Load Data
        data = scipy.io.loadmat('data/cylinder_nektar_wake.mat')

        # Unpack
        X_star = data['X_star']
        t_star = data['t']
        U_star = data['U_star']
        P_star = data['p_star']

        N = X_star.shape[0]
        T = t_star.shape[0]

        # Rearrange Data 
        XX = np.tile(X_star[:,0:1], (1,T)) # N x T
        YY = np.tile(X_star[:,1:2], (1,T)) # N x T
        TT = np.tile(t_star, (1,N)).T # N x T

        UU = U_star[:,0,:] # N x T
        VV = U_star[:,1,:] # N x T
        PP = P_star # N x T

        x = XX.flatten()[:,None] # NT x 1
        y = YY.flatten()[:,None] # NT x 1
        t = TT.flatten()[:,None] # NT x 1

        u = UU.flatten()[:,None] # NT x 1
        v = VV.flatten()[:,None] # NT x 1
        p = PP.flatten()[:,None] # NT x 1

        self.Zd_full = torch.tensor(np.concatenate((x, y, t), 1), dtype=torch.float32)
        self.Ud_full = torch.tensor(np.concatenate((u, v, p), 1), dtype=torch.float32)

        # Subsample
        idx = np.random.choice(N*T, samplesize, replace=False)
        self.Zd = self.Zd_full[idx]
        self.Ud = self.Ud_full[idx]

        # Add noise
        self.Ud += noise_level * torch.mean(abs(self.Ud)) * torch.randn_like(self.Ud)
    

    def get_quantities(self, model, t=10):

        slice_shape = (50, 100)
        lambda2 = 0.01

        x = torch.linspace(1, 8, slice_shape[1])
        y = torch.linspace(-2, 2, slice_shape[0])
        X, Y = torch.meshgrid(x, y)
        T = torch.ones_like(X) * t
        Z = torch.stack([X.T, Y.T, T.T], dim=2).reshape(-1, 3).requires_grad_(True)
        U = model.u(Z)
        U_np = U.detach().numpy().reshape(*X.shape, 2)

        ###############
        # PREDICTIONS #
        ###############

        # Predicted preassure
        pz_pred = compute_grad(U[:, 1:2], Z)
        px_pred = pz_pred[:, 0].reshape(slice_shape).detach().numpy()
        py_pred = pz_pred[:, 1].reshape(slice_shape).detach().numpy()
        p_pred = U_np[..., 1].T

        # Predicted velocity field and vorticity
        Psi_Z = compute_grad(U[..., 0], Z)
        psi_x = Psi_Z[:, 0]
        psi_y = Psi_Z[:, 1]

        u_pred_ = psi_y
        v_pred_ = -psi_x

        u_y_pred_ = compute_grad(u_pred_, Z)[:, 1].detach().numpy().reshape(slice_shape)
        v_x_pred_ = compute_grad(v_pred_, Z)[:, 0].detach().numpy().reshape(slice_shape)

        omega_pred = v_x_pred_ - u_y_pred_
        u_pred = u_pred_.detach().numpy().reshape(slice_shape)
        v_pred = v_pred_.detach().numpy().reshape(slice_shape)

        # Predicted pressure field
        p_pred = U[:, 1:2].detach().numpy().T.reshape(slice_shape)

        # Predicted residuals
        res_pred = model.F(model.F_input(Z, U))
        res_f_pred = res_pred[:, 0:1].detach().numpy().reshape(slice_shape)
        res_g_pred = res_pred[:, 1:2].detach().numpy().reshape(slice_shape)


        ############
        # SOLUTION #
        ############

        # Mask for the current time step
        mask = self.Zd_full[:, 2] == t

        # True preassure gradients
        p_true = self.Ud_full[mask, 2].reshape(slice_shape)
        p_x_true = np.gradient(p_true, x, axis=1)
        p_y_true = np.gradient(p_true, y, axis=0)

        # True velocity field and vorticity
        u_true = self.Ud_full[mask, 0].reshape(slice_shape)
        v_true = self.Ud_full[mask, 1].reshape(slice_shape)

        u_y_true = np.gradient(u_true, y, axis=0)
        v_x_true = np.gradient(v_true, x, axis=1)
        omega_true = v_x_true - u_y_true

        # True residuals
        res_true_f = - lambda2 * (np.gradient(np.gradient(u_true, x, axis=1), x, axis=1) + np.gradient(np.gradient(u_true, y, axis=0), y, axis=0))
        res_true_g = - lambda2 * (np.gradient(np.gradient(v_true, x, axis=1), x, axis=1) + np.gradient(np.gradient(v_true, y, axis=0), y, axis=0))


        ############
        # EXPECTED #
        ############

        # Expected residuals
        u_x_exp = compute_grad(u_pred_, Z)[:, 0]
        u_y_exp = compute_grad(u_pred_, Z)[:, 1]
        v_x_exp = compute_grad(v_pred_, Z)[:, 0]
        v_y_exp = compute_grad(v_pred_, Z)[:, 1]

        u_xx_pred = compute_grad(u_x_exp, Z)[:, 0].detach().numpy().reshape(slice_shape)
        u_yy_pred = compute_grad(u_y_exp, Z)[:, 1].detach().numpy().reshape(slice_shape)
        v_xx_pred = compute_grad(v_x_exp, Z)[:, 0].detach().numpy().reshape(slice_shape)
        v_yy_pred = compute_grad(v_y_exp, Z)[:, 1].detach().numpy().reshape(slice_shape)

        res_exp_f = - lambda2 * (u_xx_pred + u_yy_pred)
        res_exp_g = - lambda2 * (v_xx_pred + v_yy_pred)







        return {
            "x": x,
            "y": y,
            "u_pred": u_pred,
            "v_pred": v_pred,
            "p_pred": p_pred,
            "px_pred": px_pred,
            "py_pred": py_pred,
            "omega_pred": omega_pred,
            "f_res_pred": res_f_pred,
            "g_res_pred": res_g_pred,
            "u_true": u_true,
            "v_true": v_true,
            "p_true": p_true,
            "px_true": p_x_true,
            "py_true": p_y_true,
            "omega_true": omega_true,
            "f_res_true": res_true_f,
            "g_res_true": res_true_g,
            "f_res_exp": res_exp_f,
            "g_res_exp": res_exp_g
        }




# import numpy as np
# import matplotlib.pyplot as plt

# # Select time step index (e.g., t=0)
# t_idx = 0

# # Extract spatial coordinates
# x, y = X_star[:, 0], X_star[:, 1]
# U_star


# plt.figure(figsize=(8, 6))
# plt.imshow(U_star[:, 0, t_idx].reshape(n_y, n_x), extent=(x.min(), x.max(), y.min(), y.max()), cmap="seismic", vmin=-1, vmax=1)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title(f"Velocity Magnitude at t={t_star[t_idx, 0]:.2f}")
# plt.colorbar()
# plt.show()


# plt.figure(figsize=(8, 6))
# plt.imshow(U_star[:, 1, t_idx].reshape(n_y, n_x), extent=(x.min(), x.max(), y.min(), y.max()), cmap="seismic", vmin=-1, vmax=1)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title(f"Velocity Magnitude at t={t_star[t_idx, 0]:.2f}")
# plt.colorbar()
# plt.show()