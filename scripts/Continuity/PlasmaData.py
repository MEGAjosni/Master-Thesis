import numpy as np
import torch


class PlasmaData:

    def __init__(self, time_slices=None, noise_level=None, shuffle=False, x_range=None, t_range=None):

        # Load dataset
        data_files = np.load('data/100134_1.8MW_55s_aaro_5_3.npz', allow_pickle=True)
        data = {key: data_files[key] for key in data_files.files}
        

        # Unpack
        prob = data['prob'][()]
        x = prob['x']
        t = prob['tout']
        x_idx = np.arange(x_range[0], x_range[1]) if x_range else np.arange(len(x))
        t_idx = np.arange(t_range[0], t_range[1]) if t_range else np.arange(len(t))
        if time_slices: t_idx = time_slices
            

        prob = data['prob'][()]
        x = prob['x']                        # Normalized poloidal flux
        D = prob['D']                        # Diffusion coefficient
        V = prob['V']                        # Pinch velocity
        S = prob['S']                        # Source term
        dVdx = prob['dVdx'].reshape(-1, 1)   # Volume expansion term
        g1 = prob['g1'].reshape(-1, 1)       # ⟨∇ρ⟩
        g2 = prob['g2'].reshape(-1, 1)       # ⟨(∇ρ)^2⟩
        bc = prob['bc'].reshape(-1, 1)       # Boundary condition at psi_n ≈ 1.106 (or 1.0 ???)
        n0 = prob['n0'].reshape(-1, 1)       # Initial density
        t = prob['tout']                     # Time
        itaupar = prob['itaupar']            # Parallel loss time

        # Solution data
        sol = data['sol'][()]
        ne = sol['ne']                       # Electron density (Found solution)

        # Convert to tensors
        X_tensor, T_tensor = torch.meshgrid(torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32))
        D_tensor = torch.tensor(D, dtype=torch.float32)
        V_tensor = torch.tensor(V, dtype=torch.float32)
        S_tensor = torch.tensor(S, dtype=torch.float32)
        dVdx_tensor = torch.repeat_interleave(torch.tensor(dVdx, dtype=torch.float32), 50, dim=1)
        g1_tensor = torch.repeat_interleave(torch.tensor(g1, dtype=torch.float32), 50, dim=1)
        g2_tensor = torch.repeat_interleave(torch.tensor(g2, dtype=torch.float32), 50, dim=1)
        itaupar_tensor = torch.tensor(itaupar, dtype=torch.float32)
        ne_tensor = torch.tensor(ne, dtype=torch.float32)

        # Save full dataset
        self.Zd_full = torch.cat((
            X_tensor.reshape(-1, 1),
            T_tensor.reshape(-1, 1)
            ),
            dim=1
        )
        self.Ud_full = torch.cat((
            D_tensor.reshape(-1, 1),
            V_tensor.reshape(-1, 1),
            S_tensor.reshape(-1, 1),
            dVdx_tensor.reshape(-1, 1),
            g1_tensor.reshape(-1, 1),
            g2_tensor.reshape(-1, 1),
            itaupar_tensor.reshape(-1, 1),
            ne_tensor.reshape(-1, 1)
            ),
            dim=1)

        # Sample data
        self.Zd = torch.cat((
            X_tensor[x_idx, t_idx].reshape(-1, 1),
            T_tensor[x_idx, t_idx].reshape(-1, 1)
            ),
            dim=1
        )
        self.Ud = torch.cat((
            D_tensor[x_idx, t_idx].reshape(-1, 1),
            V_tensor[x_idx, t_idx].reshape(-1, 1),
            S_tensor[x_idx, t_idx].reshape(-1, 1),
            dVdx_tensor[x_idx, t_idx].reshape(-1, 1),
            g1_tensor[x_idx, t_idx].reshape(-1, 1),
            g2_tensor[x_idx, t_idx].reshape(-1, 1),
            itaupar_tensor[x_idx, t_idx].reshape(-1, 1),
            ne_tensor[x_idx, t_idx].reshape(-1, 1)
            ),
            dim=1
        )

        def __len__(self):
            return len(self.Ud)
        
        def __getitem__(self, idx):
            return self.Zd[idx], self.Ud[idx]