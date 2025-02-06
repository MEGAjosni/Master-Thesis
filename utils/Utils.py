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
    Class wrapper for SINDy to make it more compatible with sklearn.
    Changes:
        Z = [t, X]
        y = X_dot
    '''
    def fit(self, Z, y, **fit_kwargs):
        return super(SINDy_sklearn, self).fit(x=Z[:, 1:], t=Z[:, 0], x_dot=y, **fit_kwargs)

    def score(self, Z, y, **score_kwargs):
        return super(SINDy_sklearn, self).score(t=Z[:, 0], x=Z[:, 1:], x_dot=y, **score_kwargs)