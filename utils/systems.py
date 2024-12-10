import torch
from abc import ABC, abstractmethod

class System:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.set_boundary_condition_points()
        self.get_data_points()
        self.get_collocation_points()

    @property
    def z_b(self):
        return self._z_b
    
    @property
    def U_b(self):
        return self._U_b
    
    @property
    def z_d(self):
        return self._z_d
    
    @property
    def U_d(self):
        return self._U_d
    
    @property
    def z_c(self):
        return self._z_c

    def set_boundary_condition_points(self):
        self._z_b = torch.empty(0, self.dim)
        self._U_b = torch.empty(0, self.out_dim)

    def get_data_points(self):
        self._z_d = torch.empty(0, self.dim)
        self._U_d = torch.empty(0, self.out_dim)

    def get_collocation_points(self):
        self._z_c = torch.empty(0, self.dim)


    @abstractmethod
    def pde_residual(self, z, U, **params):
        """This method must be implemented."""
        pass

    @abstractmethod
    


class LotkaVolterra(System):
