import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, in_channels, debug=False):
        super(ConvAE, self).__init__()
        self.in_channels = in_channels
        self.debug = debug
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, 16, (3 ,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3 ,3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
        self.t_conv2 = nn.ConvTranspose2d(32, 16, (3 ,3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(16, self.in_channels, (3 ,3), stride=1, padding=1)

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
