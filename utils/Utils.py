import torch
import numpy as np
import pysindy as ps
from sklearn.model_selection import TimeSeriesSplit
from functools import reduce
from scipy.optimize import fmin_l_bfgs_b
eps = np.finfo('double').eps

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
        return super(SINDy_sklearn, self).score(t=Z[:, 0], x=Z[:, 1:], x_dot=y, **score_kwargs)
    


# class LBFGSScipy(torch.optim.Optimizer):
#     """Wrap L-BFGS algorithm, using scipy routines.
#     .. warning::
#         This optimizer doesn't support per-parameter options and parameter
#         groups (there can be only one).
#     .. warning::
#         Right now CPU only
#     .. note::
#         This is a very memory intensive optimizer (it requires additional
#         ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
#         try reducing the history size, or use a different algorithm.
#     Arguments:
#         max_iter (int): maximal number of iterations per optimization step
#             (default: 20)
#         max_eval (int): maximal number of function evaluations per optimization
#             step (default: max_iter * 1.25).
#         tolerance_grad (float): termination tolerance on first order optimality
#             (default: 1e-5).
#         tolerance_change (float): termination tolerance on function
#             value/parameter changes (default: 1e-9).
#         history_size (int): update history size (default: 100).
#     """

#     def __init__(self, params, max_iter=20, max_eval=None,
#                  tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
#                  ):
#         if max_eval is None:
#             max_eval = max_iter * 5 // 4
#         defaults = dict(max_iter=max_iter, max_eval=max_eval,
#                         tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
#                         history_size=history_size)
#         super(LBFGSScipy, self).__init__(params, defaults)

#         if len(self.param_groups) != 1:
#             raise ValueError("LBFGS doesn't support per-parameter options "
#                              "(parameter groups)")

#         self._params = self.param_groups[0]['params']
#         self._numel_cache = None

#         self._n_iter = 0
#         self._last_loss = None

#     def _numel(self):
#         if self._numel_cache is None:
#             self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
#         return self._numel_cache

#     def _gather_flat_grad(self):
#         views = []
#         for p in self._params:
#             if p.grad is None:
#                 view = p.data.new(p.data.numel()).zero_()
#             elif p.grad.data.is_sparse:
#                 view = p.grad.data.to_dense().view(-1)
#             else:
#                 view = p.grad.data.view(-1)
#             views.append(view)
#         return torch.cat(views, 0)

#     def _gather_flat_params(self):
#         views = []
#         for p in self._params:
#             if p.data.is_sparse:
#                 view = p.data.to_dense().view(-1)
#             else:
#                 view = p.data.view(-1)
#             views.append(view)
#         return torch.cat(views, 0)

#     def _distribute_flat_params(self, params):
#         offset = 0
#         for p in self._params:
#             numel = p.numel()
#             # view as to avoid deprecated pointwise semantics
#             p.data = params[offset:offset + numel].view_as(p.data)
#             offset += numel
#         assert offset == self._numel()

#     def step(self, closure):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         assert len(self.param_groups) == 1

#         group = self.param_groups[0]
#         max_iter = group['max_iter']
#         max_eval = group['max_eval']
#         tolerance_grad = group['tolerance_grad']
#         tolerance_change = group['tolerance_change']
#         history_size = group['history_size']

#         def wrapped_closure(flat_params):
#             """closure must call zero_grad() and backward()"""
#             flat_params = torch.from_numpy(flat_params)
#             self._distribute_flat_params(flat_params)
#             loss = closure()
#             self._last_loss = loss
#             loss = loss.item()
#             flat_grad = self._gather_flat_grad().numpy()
#             return loss, flat_grad

#         def callback(flat_params):
#             self._n_iter += 1
#             # print('Iter %i Loss %.5f' % (self._n_iter, self._last_loss.data[0]))

#         initial_params = self._gather_flat_params()

#         fmin_l_bfgs_b(wrapped_closure, initial_params, maxiter=max_iter,
#                       maxfun=max_eval,
#                       factr=tolerance_change / eps, pgtol=tolerance_grad, epsilon=0,
#                       m=history_size,
#                       callback=callback)
        
#         return self._last_loss