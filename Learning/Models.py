import torch
from torch import nn
import numpy as np

def weights_init(m, mu=-1., std=0.01):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=mu, std=std)
        
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         xavier(m.bias.data)

# Linear features
class LinearRewardModel_Tanh(nn.Module):
    
    def __init__(self, phi_dim, out_bias=0., gain=1., steep=1.):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.out_bias = out_bias
        self.steep = steep
        self.gain = gain
        
    def forward(self, phi):
        # Force R to be non-positive, use bias to shift level
        return self.gain * self.tanh(self.w(phi) * self.steep) + self.out_bias

class LinearRewardModel_ReLU(nn.Module):
    
    def __init__(self, phi_dim, out_bias=0., gain=1., steep=1., invert=True):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.out_bias = out_bias
        self.steep = steep
        self.gain = gain
        self.sign = -1. if invert else 1.
        
    def forward(self, phi):
        # Force R to be non-positive, use bias to shift level
        return self.sign * self.gain * self.relu(self.w(phi) * self.steep) + self.out_bias
    
class LinearRewardModel_Squared_ReLU(nn.Module):
    
    def __init__(self, phi_dim, out_bias=0., gain=1., steep=1., invert=True):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.out_bias = out_bias
        self.steep = steep
        self.gain = gain
        self.sign = -1. if invert else 1.
        
    def forward(self, phi):
        # Force R to be non-positive, use bias to shift level
        return self.sign * self.gain * self.relu((self.w(phi)**2) * self.steep) + self.out_bias
