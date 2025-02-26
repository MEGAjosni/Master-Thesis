import torch
import numpy as np
import pysindy as ps
from sklearn.model_selection import TimeSeriesSplit

# Disable Warning for sindy
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

class SoftAdapt(torch.nn.Module):
    def __init__(self, beta=0.0, loss_weigthed=False):
        super(SoftAdapt, self).__init__()
        self.beta = beta
        self.loss_weigthed = loss_weigthed
        self.prev_losses = 0.0

    def forward(self, losses):
        self.prev_losses = losses
        s = losses - self.prev_losses
        t = torch.exp(self.beta*s)
        if self.loss_weigthed:
            return losses*t / torch.sum(losses*t)
        else:
            return t / torch.sum(t)


class DeadZoneLinear(torch.nn.Module):
    def __init__(self, a=0.1):
        super(DeadZoneLinear, self).__init__()
        self.a = a  # The range [-a, a] where the function outputs 0

    def forward(self, x):
        return torch.where(x > self.a, x - self.a, 
                           torch.where(x < -self.a, x + self.a, torch.tensor(0.0, device=x.device)))


def RAD_sampler(candidate_points, residuals, N_RAD, k=1.0, c=1.0):
    residuals = torch.abs(residuals)
    error_eq = torch.pow(residuals, k) / torch.pow(residuals, k).mean() + c
    err_eq_normalized = error_eq / error_eq.sum()
    idx = torch.multinomial(err_eq_normalized.squeeze(), N_RAD, replacement=False)

    return candidate_points[idx]


def sample_collocation_points(N, n_dims, lb, ub, method='sobol', verbose=False):
    """
    Sample N collocation points from the domain [lb, ub]
    """

    if method == 'sobol':
        sobol = torch.quasirandom.SobolEngine(dimension=n_dims)
        Zc = sobol.draw(n=N, dtype=torch.float32)
    
    elif method == 'lhs':
        hypercube = qmc.LatinHypercube(d=n_dims)
        Zc = torch.tensor(hypercube.random(n=N), dtype=torch.float32)
    
    elif method == 'uniform':
        Zc = torch.rand(N, n_dims)
    
    elif method == 'grid':
        # Will not generate N points if N is not a perfect square
        root = N**(1/n_dims)
        m = int(np.ceil(N**(1/n_dims)))
        print(f"[Info]: The {n_dims}-root of {N} is not an integer, sample will contain {m**n_dims} points instead.") if root % 1 != 0 and verbose else None
        grid = torch.linspace(0, 1, m)
        Zc = torch.cartesian_prod(*[grid]*n_dims)
        if n_dims == 1:
            Zc = Zc.unsqueeze(1)

    else:
        raise ValueError(f"Invalid sample method: {method}\nChoose from ['sobol', 'lhs', 'uniform', 'grid']")

    
    # Scale the points to the domain [lb, ub]
    lb = torch.tensor(lb)
    ub = torch.tensor(ub)
    Zc = lb + (ub - lb) * Zc

    return Zc




# # # # # # # # # # # # # # # # # # # #
#                                     #
#    SSSS  III  N   N  DDDD   y   y   #
#   S       I   NN  N  D   D   y y    #
#    SSS    I   N N N  D   D    y     #
#       S   I   N  NN  D   D    y     #
#   SSSS   III  N   N  DDDD     y     #
#                                     #
# # # # # # # # # # # # # # # # # # # #


class SINDy_sklearn(ps.SINDy):
    '''
    pysindy.SINDy class wrapper for better sklearn compatibility.
    Variables are renamed to match sklearn's fit and score methods.
        Z = [t, X]
        y = X_dot
    '''
    def fit(self, Z, y, **fit_kwargs):
        return super(SINDy_sklearn, self).fit(x=Z[:, 1:], t=Z[:, 0], x_dot=y, **fit_kwargs)

    def score(self, Z, y, **score_kwargs):
        return super(SINDy_sklearn, self).score(x=Z[:, 1:], t=Z[:, 0], x_dot=y, **score_kwargs)
    
