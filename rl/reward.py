import numpy as np
import torch, torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, input_shape, debug=False):
        super(ConvAE, self).__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]
        self.debug = debug
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, 16, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 1) # calculated for 32x32 input
        self.sigmoid = nn.Sigmoid()

        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
        self.t_conv2 = nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(16, self.in_channels, (3, 3), stride=1, padding=1)

    def encode(self, x, debug=False):
        debug = True if self.debug else debug
        if debug: print(x.shape)
        x = torch.relu(self.conv1(x))
        if debug: print(x.shape)
        x = self.pool(x)
        if debug: print(x.shape)
        x = torch.relu(self.conv2(x))
        if debug: print(x.shape)
        return x

    def decode(self, x, debug=True):
        debug = True if self.debug else debug
        if debug: print(x.shape)
        x = torch.relu(self.t_conv2(x))
        if debug: print(x.shape)
        x = torch.sigmoid(self.t_conv1(x))
        if debug: print(x.shape)
        return x

    def forward(self, x):
        x_enc = self.encode(x)
        x_ = self.decode(x_enc)
        return x_

    def reward(self, x):
        
        x = self.encode(x)
        x = x.view(-1, np.product(x.size()[1:]))
        x = self.fc1(x)
        x = -self.sigmoid(x)
        return x
