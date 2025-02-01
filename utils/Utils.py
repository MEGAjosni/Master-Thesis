import torch
import numpy as np
import pysindy as ps

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
    

def cv_sindy(X, t, X_dot, degree=2, taus=np.linspace(0.1, 1.0, 100), n_splits=5, feature_names=None):
    """
    Performs time series cross-validation to find the best tau for SINDy model selection.

    Parameters:
    - X: np.ndarray, input data
    - t: np.ndarray, time points
    - X_dot: np.ndarray, derivative of X with respect to time
    - degree: int, polynomial degree for SINDy library (default: 2)
    - taus: np.ndarray, range of tau values to test (default: np.linspace(0.1, 1.0, 100))
    - n_splits: int, number of time series cross-validation splits (default: 5)
    - feature_names: list, names of features for SINDy model (default: ['x', 'y'])

    Returns:
    - final_model: trained SINDy model with the best tau
    - best_tau: float, best tau value found
    """

    # Define SINDy library
    lib = ps.PolynomialLibrary(degree=degree)

    # Define cross-validation split size
    split_size = len(X) // (n_splits + 1)  # Ensures valid train-test splits

    mean_scores = np.zeros_like(taus)

    warnings.filterwarnings("ignore", category=UserWarning)
    for i, tau in enumerate(taus):
        scores = []  # Store scores for each time split

        # Perform time series cross-validation
        for split in range(n_splits):
            train_end = split_size * (split + 1)
            test_start = train_end
            test_end = test_start + split_size

            # Define train and test indices
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, len(X)))

            # Initialize the SINDy model with the current tau
            model = ps.SINDy(
                feature_library=lib,
                optimizer=ps.STLSQ(threshold=tau),
                feature_names=feature_names
            )

            # Fit the model on the training set
            model.fit(x=X[train_idx], t=t[train_idx], x_dot=X_dot[train_idx])

            # Score the model on the testing set
            scores.append(model.score(X[test_idx], t=t[test_idx], x_dot=X_dot[test_idx]))

        # Compute the mean score across all splits for this tau
        mean_scores[i] = np.mean(scores)

    warnings.filterwarnings("default", category=UserWarning)

    # Find the best tau
    best_tau = taus[np.argmax(mean_scores)]

    # Train the final model with the best tau
    final_model = ps.SINDy(
        feature_library=lib,
        optimizer=ps.STLSQ(threshold=best_tau),
        feature_names=feature_names
    )

    final_model.fit(x=X, t=t, x_dot=X_dot)

    return final_model, best_tau
   
