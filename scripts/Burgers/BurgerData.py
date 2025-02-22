import numpy as np
import torch

class BurgerData:
    def __init__(self, time_slices=None, noise_level=None, shuffle=False):
        
        # Load dataset
        data = np.load('C:/Users/jonas/OneDrive - Danmarks Tekniske Universitet/Master Thesis/Master-Thesis/scripts/Burgers/data/Burgers.npz')
        tsol, xsol, usol = data['t'], data['x'], data['usol']
        self.tsol, self.xsol, self.usol = tsol, xsol, usol
        T, X = np.meshgrid(tsol, xsol)
        self.original_shape = usol.shape

        # Save full dataset
        Zd_full = torch.tensor(np.vstack([T.flatten(), X.flatten()]).T, dtype=torch.float32)
        Ud_full = torch.tensor(usol.flatten(), dtype=torch.float32).reshape(-1, 1)
        self.data_points_full = Zd_full, Ud_full
        self.t, self.x = tsol, xsol

        # Sample data
        if time_slices:
            T, X = T[:, time_slices], X[:, time_slices]
            usol = usol[:, time_slices]

        t, x = T.flatten(), X.flatten()
        Zd = torch.tensor(np.vstack([t, x]).T, dtype=torch.float32)
        Ud = torch.tensor(usol.flatten(), dtype=torch.float32).reshape(-1, 1)

        self.data_points_noiseless = torch.clone(Zd), torch.clone(Ud)

        # Add noise
        if noise_level:
            Ud += noise_level * torch.mean(abs(Ud)) * torch.randn(*Ud.shape)

        # Shuffle data
        if shuffle:
            idx = torch.randperm(Zd.shape[0])
            Zd, Ud = Zd[idx], Ud[idx]
        
        self.data_points = Zd, Ud

    def __len__(self):
        return len(self.Ud)
    
    def __getitem__(self, idx):
        return self.Zd[idx], self.Ud[idx]