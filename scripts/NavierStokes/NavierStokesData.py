import scipy.io
import numpy as np
import torch

class NavierStokesData:
    def __init__(self, samplesize=5000):

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