import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim


class RewardConvAE(nn.Module):
    def __init__(self, input_shape, lr=0.1, weight_decay=0, dropout_prob=0., debug=False):

        super(RewardConvAE, self).__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]
        self.debug = debug
        self.dropout_p = dropout_prob
#         # Input: 28 x 28
#         self.fc1_in = 3136
#         self.fc1_in_shape = (16, 14, 14)
        # Input: 32 x 32
        self.fc1_in = 4096
        self.fc1_in_shape = (16, 16, 16)
        
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, 8, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(8, 16, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc1_in, 128, bias=True)
        self.fc2 = nn.Linear(128, 32, bias=True) # TODO: eval use of bias
        self.dropout = nn.Dropout(self.dropout_p, inplace=False) # TODO: eval use of dropout in feature encoding
        self.sigmoid = nn.Sigmoid()
        self.fc_reward = nn.Linear(32, 1, bias=False)

        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
        self.t_fc2 = nn.Linear(32, 128, bias=True)
        self.t_fc1 = nn.Linear(128, self.fc1_in, bias=True)
        self.t_conv2 = nn.ConvTranspose2d(16, 8, (3, 3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(8, self.in_channels, (3, 3), stride=1, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
    def encode(self, x, debug=False):
        debug = True if self.debug else debug
        if debug: print(x.shape)
        x = torch.relu(self.conv1(x))
        if debug: print(x.shape)
        x = self.pool(x)
        if debug: print(x.shape)
        x = torch.relu(self.conv2(x))
        if debug: print(x.shape)
        x = x.view(-1, np.product(x.size()[1:]))
        if debug: print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        if debug: print(x.shape)
        x = self.fc2(x)
        if debug: print(x.shape)
        return x

    def decode(self, x, debug=False):
        debug = True if self.debug else debug
        if debug: print(x.shape)
        x = torch.relu(self.t_fc2(x))
        if debug: print(x.shape)
        x = self.dropout(x)
        if debug: print(x.shape)
        x = torch.relu(self.t_fc1(x))
        if debug: print(x.shape)
        x = x.view(-1, *self.fc1_in_shape)
        if debug: print(x.shape)
        x = torch.relu(self.t_conv2(x))
        if debug: print(x.shape)
        x = self.sigmoid(self.t_conv1(x))
        if debug: print(x.shape)
        return x

    def forward(self, x):
        x_enc = self.encode(x)
        x_ = self.decode(x_enc)
        return x_

    def reward(self, x):
        x = self.encode(x)
        x = self.fc_reward(x)
        x = -self.sigmoid(x)
        return x

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)


class RewardLinear(nn.Module):
    
    def __init__(self, phi_dim, lr=0.1, weight_decay=0, debug=False):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, phi):
        r_logit = self.w(phi)
        return r_logit

    def reward(self, phi):
        return -self.sigmoid(self.forward(phi))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)
