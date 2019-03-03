import torch
from torch import nn
import numpy as np

# Linear features
class LinearRewardModel_Tanh(nn.Module):
    
    def __init__(self, phi_dim, const_bias=0.):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.const_bias = const_bias
        
    def forward(self, phi):
        return self.tanh(self.w(phi)) + self.const_bias # Force R to be non-positive
    
    
class LinearRewardModel_ReLU(nn.Module):
    
    def __init__(self, phi_dim, const_bias=0.):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.const_bias = const_bias
        
    def forward(self, phi):
        return -self.relu(-self.w(phi)) + self.const_bias # Force R to be non-positive
    
