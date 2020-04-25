import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim


class PyTorchNNModuleAug(nn.Module):
    """
    Class augmenting nn.Module.
    """
    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__()
        self.input_shape = input_shape
        if (isinstance(input_shape, list) or isinstance(input_shape, tuple)) and len(input_shape) == 3:
            self.in_channels = self.input_shape[0]
        else:
            self.in_channels = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.debug = debug
        self.optimizer = None

    def in_shape(self):
        return self.input_shape

    def in_channels(self):
        return self.in_channels

    def set_optimizer(self, parameters=None):
        if parameters is None:
            parameters = self.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

    def zero_grad(self):
        if self.optimizer is None:
            raise Exception("Optimizer is not setup!")
        self.optimizer.zero_grad()

    def step(self):
        if self.optimizer is None:
            raise Exception("Optimizer is not setup!")
        self.optimizer.step()


class Encoder(PyTorchNNModuleAug):
    """
    Abstract Class for Encoder.
    """
    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)

    def initialize(self):
        self.setup_encoder_layers()
        self.set_optimizer()

    def setup_encoder_layers(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.encode(x)


class Decoder(PyTorchNNModuleAug):
    """
    Abstract Class for Decoder.
    """
    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)

    def initialize(self):
        self.setup_decoder_layers()
        self.set_optimizer()

    def setup_decoder_layers(self):
        raise NotImplementedError

    def decode(self, x_enc):
        raise NotImplementedError

    def forward(self, x_enc):
        return self.decode(x_enc)


class AutoEncoder(Encoder, Decoder):
    """
    Abstract Class for AutoEncoder.
    """
    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)

    def initialize(self):
        self.setup_encoder_layers()
        self.setup_decoder_layers()
        self.set_optimizer()

    def forward(self, x):
        x_enc = self.encode(x)
        x_ = self.decode(x_enc)
        return x_



class ConvFCAutoEncoder(AutoEncoder):
    """
    Class for Convolutional AutoEncoder with FC layers.
    """
    def __init__(self, input_shape, fc1_in=4096, lr=0.1, weight_decay=0, dropout_prob=0., debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.dropout_p = dropout_prob
        self.fc1_in = fc1_in
        self.c1 = 8
        self.fc1_in_shape = None
        self.initialize()

    def setup_encoder_layers(self):
        c1 = self.c1
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, c1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(c1, c1 * 2, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc1_in, c1 * 16, bias=True)
        self.fc2 = nn.Linear(c1 * 16, c1 * 8, bias=True)
        self.dropout = nn.Dropout(self.dropout_p, inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.fc_reward = nn.Linear(c1 * 8, 1, bias=False)

    def setup_decoder_layers(self):
        c1 = self.c1
        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
        self.t_fc2 = nn.Linear(c1 * 8, c1 * 16, bias=True)
        self.t_fc1 = nn.Linear(c1 * 16, self.fc1_in, bias=True)
        self.t_conv2 = nn.ConvTranspose2d(c1 * 2, c1, (3, 3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(c1, self.in_channels, (3, 3), stride=1, padding=1)

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

        self.fc1_in_shape = x[0].shape

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

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        pr = -self.sigmoid(r)
        if return_latent:
            return pr, z
        else:
            return pr

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)


class ConvAutoEncoder(AutoEncoder):
    """
    Class for Fully Convolutional AutoEncoder.
    """
    def __init__(self, input_shape, z_dim=7500, lr=0.1, weight_decay=0, dropout_prob=0., debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.dropout_p = dropout_prob
        self.z_dim = z_dim
        self.c1 = 8
        self.initialize()

    def setup_encoder_layers(self):
        c1 = self.c1
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(self.in_channels, c1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(c1, c1 * 2, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.conv1x1 = nn.Conv2d(c1 * 2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc_reward = nn.Linear(self.z_dim, 1, bias=False)

    def setup_decoder_layers(self):
        c1 = self.c1
        # Decoder layers
        self.t_conv1x1 = nn.Conv2d(1, c1 * 2, 1)
        self.t_conv2 = nn.ConvTranspose2d(c1 * 2, c1, (3, 3), stride=2, padding=1, output_padding=1)
        self.t_conv1 = nn.ConvTranspose2d(c1, self.in_channels, (3, 3), stride=1, padding=1)

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
        x = self.flatten(x)
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

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        pr = -self.sigmoid(r)
        if return_latent:
            return pr, z
        else:
            return pr

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)


class RewardLinear(Encoder):
    """
    Class for Linear Reward Model.
    """
    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def init_const(self, value=1.):
        torch.nn.init.constant_(self.w.weight.data, value)

    def init_uniform(self, lo=-1, hi=1):
        torch.nn.init.uniform_(self.w.weight.data, lo, hi)

    def setup_encoder_layers(self):
        if isinstance(self.in_shape(), list) or isinstance(self.in_shape(), tuple):
            assert len(self.in_shape()) == 1
            phi_dim = self.in_shape()[0]
        elif isinstance(self.in_shape(), int):
            phi_dim = self.in_shape()
        else:
            raise Exception("Can't handle input_shape {}!".format(self.in_shape()))
        self.fc_reward = nn.Linear(phi_dim, 1, bias=False)

    def encode(self, x, debug=False):
        return x

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        pr = -self.sigmoid(r)
        if return_latent:
            return pr, z
        else:
            return pr

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)