import numpy as np
# Torch
import torch
from torch import nn

# chainerrl
# from chainerrl.functions.mellowmax import maximum_entropy_mellowmax
# from chainerrl.functions.mellowmax import mellowmax

def log_boltzmann_dist(Q, temperature):
    """
    PyTorch softmax implementation seems stable, but log of softmax is not.
    So log of boltzmann distribution is used.
    PyTorch Softmax Note:
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    return nn.LogSoftmax(dim=0)(Q/temperature)

BoltzmannLog = lambda Q, boltzmann_temp: log_boltzmann_dist(Q, boltzmann_temp)
Boltzmann = lambda Q, boltzmann_temp: torch.exp(log_boltzmann_dist(Q, boltzmann_temp))
GreedyLog = lambda Q: log_boltzmann_dist(Q, 1e-10)
Greedy = lambda Q: torch.exp(log_boltzmann_dist(Q, 1e-10))

# TODO: mellowmax policy
# computes only values, we also need Q distribution for policy
# Mellowmax = lambda Q, w, axis=0: (torch.logsumexp(w * Q, axis=axis)-np.log(Q.shape[axis]))/ w
# Mellowmax = lambda Q, omega: maximum_entropy_mellowmax(np.asarray(Q), omega=omega)