from abc import abstractmethod
import torch

class BVP:

    def __init__(self, params):
        """
        Boundary Value Problem object: Can evaluate the PDE residual and boundary residual.
        params: dict
            Parameter values of the BVP. If some parameters are to be estimated, they should be torch.nn.Parameter. As an example:
            params = {
                alpha: torch.nn.Parameter(torch.tensor(1.0)),
                beta: 2.3,
                }
            This will estimate alpha from an initial guess of 1.0, while beta will be kept constant at 2.3.
        """
        self.params = params

    def to(self, device):
        # Move all attributes to device
        for attr in self.__dict__.keys():
            if torch.is_tensor(getattr(self, attr)):
                setattr(self, attr, getattr(self, attr).to(device))

    @abstractmethod
    def f(self, z, U):
        """
        z: torch.Tensor
            Input points where the PDE residual is to be evaluated.

        U: torch.Tensor
            Solution (or estimate) of the PDE at the input points z. Should be the output of the solution network u(z).

        params: dict
            Parameter values of the BVP.
        """

        # This method must be implemented
        # It should return the PDE residual
        pass

    @abstractmethod
    def g(self, z, U):
        """
        z: torch.Tensor
            Input points where the boundary residual is to be evaluated.

        U: torch.Tensor
            Solution (or estimate) of the PDE at the input points z. Should be the output of the solution network u(z).

        params: dict
            Parameter values of the BVP.
        """
        # This method must be implemented
        # It should return the boundary residual
        # If there are both Dirichlet and Neumann boundary conditions, stack them on top of each other
        pass  

