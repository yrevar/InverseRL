import torch
from torch import nn
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=-1, std=0.01)
        
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         xavier(m.bias.data)

# Linear features
class LinearRewardModel_Tanh(nn.Module):
    
    def __init__(self, phi_dim, out_bias=0.):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.out_bias = out_bias
        
    def forward(self, phi):
        # Force R to be non-positive, use bias to shift level
        return self.tanh(self.w(phi)) + self.out_bias
    
    
class LinearRewardModel_ReLU(nn.Module):
    
    def __init__(self, phi_dim, out_bias=0.):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.out_bias = out_bias
        
    def forward(self, phi):
        # Force R to be non-positive, use bias to shift level
        return -self.relu(-self.w(phi)) + self.out_bias
