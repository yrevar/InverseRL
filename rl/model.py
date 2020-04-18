import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim


class Encoder(nn.Module):
    """
    Class for encoding to latent representation.
    """
    def __init__(self, encoder_fn=None):
        super(Encoder, self).__init__()
        # default transformation = identity
        self.encoder_fn = encoder_fn if encoder_fn else lambda x: x

    def encode(self, x):
        return self.encoder_fn(x)

    def __call__(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    """
    Class for decoding from latent representation.
    """
    def __init__(self, output_shape, z_dim):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.z_dim = z_dim

    def decode(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.decode(*args, **kwargs)





class RewardConvAE(nn.Module):
    def __init__(self, input_shape, z_dim=7500, lr=0.1, weight_decay=0, dropout_prob=0., debug=False):

        super(RewardConvAE, self).__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]
        self.debug = debug
        self.dropout_p = dropout_prob
#         # Input: 28 x 28
#         self.fc1_in = 3136
#         self.fc1_in_shape = (16, 14, 14)
        # Input: 32 x 32
#         self.fc1_in = fc1_in
#         self.fc1_in_shape = (16, 16, 16)
        c1 = 8
        
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, c1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(c1, c1*2, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.conv1x1 = nn.Conv2d(c1 * 2, 1, 1)
#         self.fc1 = nn.Linear(self.fc1_in, c1*16, bias=True)
#         self.fc2 = nn.Linear(c1*16, c1*8, bias=True) # TODO: eval use of bias
#         self.dropout = nn.Dropout(self.dropout_p, inplace=False) # TODO: eval use of dropout in feature encoding
        self.sigmoid = nn.Sigmoid()
        self.fc_reward = nn.Linear(z_dim, 1, bias=False)

        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
#         self.t_fc2 = nn.Linear(c1*8, c1*16, bias=True)
#         self.t_fc1 = nn.Linear(c1*16, self.fc1_in, bias=True)
        self.t_conv1x1 = nn.Conv2d(1, c1 * 2, 1)
        self.t_conv2 = nn.ConvTranspose2d(c1 * 2, c1, (3, 3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(c1, self.in_channels, (3, 3), stride=1, padding=1)

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
        x = self.pool(x)
        if debug: print(x.shape)
        x = self.conv1x1(x)
        if debug: print(x.shape)
        return x

    def decode(self, x, debug=False):
        debug = True if self.debug else debug
        x = self.t_conv1x1(x)
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

    def reward(self, x, return_latent=False):
        z = self.encode(x)
        r = self.fc_reward(self.flatten(z))
        pr = -self.sigmoid(r)
        if return_latent:
            return pr, z
        else:
            return pr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)
    
class RewardConvFCAE(nn.Module):
    def __init__(self, input_shape, fc1_in=4096, lr=0.1, weight_decay=0, dropout_prob=0., debug=False):

        super(RewardConvFCAE, self).__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]
        self.debug = debug
        self.dropout_p = dropout_prob
#         # Input: 28 x 28
#         self.fc1_in = 3136
#         self.fc1_in_shape = (16, 14, 14)
        # Input: 32 x 32
        self.fc1_in = fc1_in
        self.fc1_in_shape = (16, 16, 16)
        c1 = 8
        
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, c1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(c1, c1*2, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc1_in, c1*16, bias=True)
        self.fc2 = nn.Linear(c1*16, c1*8, bias=True) # TODO: eval use of bias
        self.dropout = nn.Dropout(self.dropout_p, inplace=False) # TODO: eval use of dropout in feature encoding
        self.sigmoid = nn.Sigmoid()
        self.fc_reward = nn.Linear(c1*8, 1, bias=False)

        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
        self.t_fc2 = nn.Linear(c1*8, c1*16, bias=True)
        self.t_fc1 = nn.Linear(c1*16, self.fc1_in, bias=True)
        self.t_conv2 = nn.ConvTranspose2d(c1*2, c1, (3, 3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(c1, self.in_channels, (3, 3), stride=1, padding=1)

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
        x = self.pool(x)
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

    def reward(self, x, return_latent=False):
        z = self.encode(x)
        r = self.fc_reward(z)
        pr = -self.sigmoid(r)
        return pr, z

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

    def init_const(self, value=1.):
        torch.nn.init.constant_(self.w.weight.data, value)

    def init_uniform(self, lo=-1, hi=1):
        torch.nn.init.uniform_(self.w.weight.data, lo, hi)

    def forward(self, phi):
        r_logit = self.w(phi)
        return r_logit

    def reward(self, phi, return_latent=False):
        return -self.sigmoid(self.forward(phi))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)
#
# class PCA:
#
#     def __init__(self, ):