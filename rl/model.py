from abc import ABC, abstractmethod
import os, numpy as np, os.path as osp
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from matplotlib import pyplot as plt


def read_input_1d(x):
    if isinstance(x, list) or isinstance(x, tuple):
        assert len(x) == 1
        dim = x[0]
    elif isinstance(x, int):
        dim = x
    else:
        raise Exception("Can't handle input_shape {}!".format(x))
    return dim


class PyTorchNNModuleAug(nn.Module):
    """
    Class augmenting nn.Module.
    """

    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = [input_shape]
        self.input_shape = input_shape
        self.in_channels = input_shape[0] if len(input_shape) == 3 else None
        self.lr = lr
        self.weight_decay = weight_decay
        self.debug = debug
        self.optimizer = None

    def in_shape(self):
        return self.input_shape

    def in_channels(self):
        return self.in_channels

    # @staticmethod
    # @abstractmethod
    # def prepare_input(x):
    #     if len(x.shape) == 3:
    #         x = torch.FloatTensor(x).unsqueeze(0).permute(0, 3, 1, 2)
    #     else:
    #         x = torch.FloatTensor(x).permute(0, 3, 1, 2)
    #     return x

    def set_optimizer(self, parameters=None):
        parameters = parameters or self.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

    def zero_grad(self):
        if self.optimizer is None:
            raise Exception("Optimizer is not setup!")
        self.optimizer.zero_grad()

    def step(self):
        if self.optimizer is None:
            raise Exception("Optimizer is not setup!")
        self.optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Encoder(ABC, PyTorchNNModuleAug):
    """
    Abstract Class for Encoder.
    """

    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.loss_history = np.array([])
        self.initialized = False
        self.epoch = 0

    def initialize(self):
        self.setup_encoder_layers()
        self.set_optimizer()
        self.initialized = True

    @abstractmethod
    def setup_encoder_layers(self):
        return

    @abstractmethod
    def encode(self, x):
        return

    def forward(self, x):
        return self.encode(x)

    @abstractmethod
    def get_state_dict(self):
        return

    @abstractmethod
    def get_optimizer_state_dict(self):
        return

    def save(self, store_dir=None, fname="ae_state.pth"):
        if not self.initialized:
            raise Exception("Failed to store. Can only load state after being initialized.")
        store_dir = store_dir or self.store_dir
        os.makedirs(store_dir, exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.get_state_dict(),
            'optimizer_state_dict': self.get_optimizer_state_dict(),
            'loss_history': self.loss_history,
        }, osp.join(store_dir, fname))

    def load(self, store_dir=None, fname="ae_state.pth"):
        if not self.initialized:
            raise Exception("Failed to load. Can only load state after being initialized.")
        store_dir = store_dir or self.store_dir
        checkpoint = torch.load(osp.join(store_dir, fname))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss_history = checkpoint['loss_history']
        self.pre_trained_weights = True


class Decoder(ABC, PyTorchNNModuleAug):
    """
    Abstract Class for Decoder.
    """

    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)

    def initialize(self):
        self.setup_decoder_layers()
        self.set_optimizer()

    @abstractmethod
    def setup_decoder_layers(self):
        return

    @abstractmethod
    def decode(self, x_enc):
        return

    def forward(self, x_enc):
        return self.decode(x_enc)


class AutoEncoder(Encoder, Decoder):
    """
    Abstract Class for AutoEncoder.
    """

    def __init__(self, input_shape, lr=0.1, weight_decay=0, store_dir="./data/ae_store_dir", debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.store_dir = store_dir
        self.loss_history = np.array([])
        self.epoch = 0
        self.initialized = False
        self.pre_trained_weights = False

    def initialize(self):
        self.setup_encoder_layers()
        self.setup_decoder_layers()
        self.set_optimizer()
        self.initialized = True

    def forward(self, x, return_latent=False):
        x_enc = self.encode(x)
        x_ = self.decode(x_enc)
        if return_latent:
            return x_, x_enc
        else:
            return x_

    def run_training(self, data_sampler, epochs=10,
                     loss_criterion=lambda x, x_: torch.sum((x - x_) ** 2 / len(x)),
                     data_process_fn=lambda x: x,
                     plot_fn=None, gif_maker=None, x_val=None):
        # data_sampler.reset_stats()
        epoch_max = self.epoch + epochs
        self.epoch = data_sampler.curr_epoch()
        while data_sampler.curr_epoch() < epoch_max:
            x = data_sampler.next_batch()
            x_ = self.forward(x)
            loss = loss_criterion(x, x_)
            self.loss_history = np.append(self.loss_history, loss.item())
            self.epoch = data_sampler.curr_epoch()
            # plot
            if data_sampler.epoch_done():
                print('Ep: {:5d}, loss: {:.5f}'.format(data_sampler.curr_epoch(), loss.item()))
                if plot_fn is not None:
                    # plotting
                    x_val_recon, x_latent = self.forward(x_val, return_latent=True)
                    plot_fn(x_val, x_val_recon, x_latent, self.loss_history, data_sampler.get_batch_size(),
                            title="ConvAE Training, {}.".format('Ep: {:5d}, loss: {:.5f}'.format(
                                data_sampler.curr_epoch(), loss.item())))
                    if gif_maker is not None:
                        gif_maker.add_plot()
                        plt.gca().cla()
                        plt.clf()
            # gradient descent
            self.zero_grad()
            loss.backward()
            self.step()

    @abstractmethod
    def get_state_dict(self):
        return

    @abstractmethod
    def get_optimizer_state_dict(self):
        return

    def save(self, store_dir=None, fname="ae_state.pth"):
        if not self.initialized:
            raise Exception("Failed to store. Can only load state after being initialized.")
        store_dir = store_dir or self.store_dir
        os.makedirs(store_dir, exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.get_state_dict(),
            'optimizer_state_dict': self.get_optimizer_state_dict(),
            'loss_history': self.loss_history,
        }, osp.join(store_dir, fname))

    def load(self, store_dir=None, fname="ae_state.pth"):
        if not self.initialized:
            raise Exception("Failed to load. Can only load state after being initialized.")
        store_dir = store_dir or self.store_dir
        checkpoint = torch.load(osp.join(store_dir, fname))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss_history = checkpoint['loss_history']
        self.pre_trained_weights = True


class ConvFCAutoEncoder(AutoEncoder):
    """
    Class for Convolutional AutoEncoder with FC layers.
    """

    def __init__(self, input_shape, z_dim=128, lr=0.1, weight_decay=0, dropout_prob=0.,
                 c1=8, cx1=4, fx1=4, store_dir=None, debug=False):
        super().__init__(input_shape, lr, weight_decay, store_dir, debug)
        self.dropout_p = dropout_prob
        self.z_dim = z_dim
        self.c1 = c1
        self.cx1 = cx1
        self.fx1 = fx1
        self.fc1_in_shape = None
        self.reward_activation_type = "sigmoid"
        self.initialize()

    def setup_encoder_layers(self):
        cin, c1, cx1, fx1, z_dim = self.in_channels, self.c1, self.cx1, self.fx1, self.z_dim
        # Encoder layers
        # Channels 1 -> 16, 16 3x3 kernels
        self.conv1 = nn.Conv2d(cin, c1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(c1, c1 * cx1, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2, 0, return_indices=True)
        self.pool_idxs = None
        self.flatten = nn.Flatten()
        # is there any easier way?
        self.fc1_in_shape, self.fc1_in = self.encode(torch.rand(1, *self.input_shape), ret_fc_shape_only=True)
        self.fc1 = nn.Linear(self.fc1_in, z_dim * fx1, bias=True)  # should this be c1 * self.z_dim * k?
        self.fc2 = nn.Linear(z_dim * fx1, z_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_p, inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.log_sigmoid = nn.LogSigmoid()
        self.reward_activation = self.sigmoid
        self.fc_reward = nn.Linear(z_dim, 1, bias=False)

    def setup_decoder_layers(self):
        cin, c1, cx1, fx1, z_dim = self.in_channels, self.c1, self.cx1, self.fx1, self.z_dim
        # Decoder layers
        # Channels 1 -> 32, 32 1x1 kernels
        self.t_fc2 = nn.Linear(z_dim, z_dim * fx1, bias=True)
        self.t_fc1 = nn.Linear(z_dim * fx1, self.fc1_in, bias=True)
        self.t_conv2 = nn.ConvTranspose2d(c1 * cx1, c1, (3, 3), stride=1, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(c1, cin, (3, 3), stride=1, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2, 0)

    def set_reward_activation(self, type="relu"):
        assert type in ["relu", "sigmoid", "log_sigmoid", "none"]
        self.reward_activation_type = type
        if type == "relu":
            self.reward_activation = self.relu
        elif type == "sigmoid":
            self.reward_activation = self.sigmoid
        elif type == "none":
            self.reward_activation = None
        else:
            self.reward_activation = self.log_sigmoid

    def encode(self, x, debug=False, ret_pool_idxs=False, ret_fc_shape_only=False):
        debug = True if self.debug else debug
        pool_idxs = []

        # prepare input
        # x = self.prepare_input(x)
        if debug: print("Encoding Input: ", x.shape)
        # Conv1-pool1
        x = torch.relu(self.conv1(x))
        if debug: print("\tConv1: ", x.shape)
        x, idxs = self.pool(x)
        pool_idxs.append(idxs)
        if debug: print("\tPool1: ", x.shape)
        # Conv2-pool2
        x = torch.relu(self.conv2(x))
        if debug: print("\tConv2: ", x.shape)
        x, idxs = self.pool(x)
        pool_idxs.append(idxs)
        if debug: print("\tPool2: ", x.shape)
        # fc1
        # self.fc1_in_shape = x[0].shape
        if ret_fc_shape_only:
            return x[0].shape, np.product(x.size()[1:])
        x = x.view(-1, np.product(x.size()[1:]))
        if debug: print("\tReshape: ", x.shape)
        x = torch.relu(self.fc1(x))
        if debug: print("\tFc1: ", x.shape)
        # fc1-dropout
        x = self.dropout(x)
        if debug: print("\tDropout1: ", x.shape)
        # fc2
        x = self.fc2(x)
        if debug: print("\tFc2: ", x.shape)

        self.pool_idxs = pool_idxs
        if ret_pool_idxs:
            return x, pool_idxs
        else:
            return x

    def bottleneck_grads(self):
        return self.fc_reward.weight.grad

    def decode(self, x, debug=False, pool_idxs=None):
        debug = True if self.debug else debug
        if pool_idxs is None:
            pool_idxs = self.pool_idxs
        if pool_idxs is None:
            _, pool_idxs = self.encode(torch.rand(1, *self.input_shape), ret_pool_idxs=True)
        if debug: print("Decoding Input: ", x.shape)
        # un-fc2
        x = torch.relu(self.t_fc2(x))
        if debug: print("\tUn-FC2: ", x.shape)
        # un-fc1-dropout
        x = self.dropout(x)
        if debug: print("\tUn-Dropout1: ", x.shape)
        # un-fc1
        x = torch.relu(self.t_fc1(x))
        if debug: print("\tUn-FC1: ", x.shape)

        # un-conv2-pool2
        x = x.view(-1, *self.fc1_in_shape)
        if debug: print("\tReshape: ", x.shape)
        x = self.unpool(x, pool_idxs.pop())
        if debug: print("\tUn-Pool2: ", x.shape)
        # un-conv2
        x = torch.relu(self.t_conv2(x))
        if debug: print("\tUn-Conv2: ", x.shape)
        # un-conv1-pool1
        x = self.unpool(x, pool_idxs.pop())
        if debug: print("\tUn-Pool1: ", x.shape)
        # un-conv1
        x = self.sigmoid(self.t_conv1(x))
        if debug: print("\tSigmoid + Un-Conv1: ", x.shape)
        return x

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        if self.reward_activation_type in ["relu", "sigmoid"]:
            r = -self.reward_activation(r)
        elif self.reward_activation_type == "none":
            pass
        else:
            r = self.reward_activation(r)
        if return_latent:
            return r, z
        else:
            return r

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)

    def get_state_dict(self):
        return self.state_dict()

    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()


class ConvAutoEncoder(AutoEncoder):
    """
    Class for Fully Convolutional AutoEncoder.
    """

    def __init__(self, input_shape, z_dim=7500, lr=0.1, weight_decay=0, dropout_prob=0., debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.dropout_p = dropout_prob
        self.z_dim = z_dim
        self.c1 = 8
        self.reward_activation_type = "sigmoid"
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
        # x = self.prepare_input(x)
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

    def set_reward_activation(self, type="relu"):
        assert type in ["relu", "sigmoid", "log_sigmoid", "none"]
        self.reward_activation_type = type
        if type == "relu":
            self.reward_activation = self.relu
        elif type == "sigmoid":
            self.reward_activation = self.sigmoid
        elif type == "none":
            self.reward_activation = None
        else:
            self.reward_activation = self.log_sigmoid

    # def reward(self, x, return_latent=False, debug=False):
    #     z = self.encode(x, debug=debug)
    #     r = self.fc_reward(z)
    #     pr = -self.sigmoid(r)
    #     if return_latent:
    #         return pr, z
    #     else:
    #         return pr

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        if self.reward_activation_type in ["relu", "sigmoid"]:
            r = -self.reward_activation(r)
        elif self.reward_activation_type == "none":
            pass
        else:
            r = self.reward_activation(r)
        if return_latent:
            return r, z
        else:
            return r

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)

    def get_state_dict(self):
        return self.state_dict()

    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def bottleneck_grads(self):
        return self.fc_reward.weight.grad


class RewardLinear(Encoder):
    """
    Class for Linear Reward Model.
    """

    def __init__(self, input_shape, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.reward_activation_type = "sigmoid"
        self.initialize()

    def init_const(self, value=1.):
        torch.nn.init.constant_(self.fc_reward.weight.data, value)

    def init_uniform(self, lo=-1, hi=1):
        torch.nn.init.uniform_(self.fc_reward.weight.data, lo, hi)

    def setup_encoder_layers(self):
        input_dim = read_input_1d(self.in_shape())
        self.fc_reward = nn.Linear(input_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reward_activation = self.sigmoid

    def set_reward_activation(self, type="relu"):
        self.reward_activation_type = type
        assert type in ["relu", "sigmoid", "log_sigmoid"]
        if type == "relu":
            self.reward_activation = self.relu
        elif type == "sigmoid":
            self.reward_activation = self.sigmoid
        else:
            self.reward_activation = self.log_sigmoid

    # @staticmethod
    # def prepare_input(x):
    #     x = torch.FloatTensor(x)
    #     return x

    def encode(self, x, debug=False):
        # x = self.prepare_input(x)
        return x

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        if self.reward_activation_type in ["relu", "sigmoid"]:
            r = -self.reward_activation(r)
        else:
            r = self.reward_activation(r)
        if return_latent:
            return r, z
        else:
            return r

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)

    def get_state_dict(self):
        return self.state_dict()

    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def bottleneck_grads(self):
        return self.fc_reward.weight.grad


class LinearUnit(nn.Module):
    """
    Class for Linear Unit.
    """

    def __init__(self, input_dim, output_dim, bias=False, lr=0.1, weight_decay=0, debug=False):
        super(LinearUnit, self).__init__()
        assert isinstance(input_dim, int) and isinstance(output_dim, int)
        self.input_dim, self.bias, self.output_dim = input_dim, bias, output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.debug = debug
        self.optimizer = None
        self.initialize()

    def _validate_dim(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            assert len(x) == 1
        elif isinstance(x, int):
            pass
        else:
            raise Exception("Can't handle dim {}!".format(x))

    def _validate_inputs(self):
        self._validate_dim(self.input_dim)
        self._validate_dim(self.output_dim)

    def init_const(self, value=1.):
        torch.nn.init.constant_(self.fc.weight.data, value)

    def init_uniform(self, lo=-1, hi=1):
        torch.nn.init.uniform_(self.fc.weight.data, lo, hi)

    def initialize(self):
        self._validate_inputs()
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=self.bias)
        self.set_optimizer()

    def set_optimizer(self, parameters=None):
        parameters = parameters or self.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

    def zero_grad(self):
        if self.optimizer is None:
            raise Exception("Optimizer is not setup!")
        self.optimizer.zero_grad()

    def step(self):
        if self.optimizer is None:
            raise Exception("Optimizer is not setup!")
        self.optimizer.step()

    # @staticmethod
    # def prepare_input(x):
    #     x = torch.FloatTensor(x)
    #     return x

    def forward(self, x, debug=False):
        # x = self.prepare_input(x)
        y = self.fc(x)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RewardMLP(Encoder):
    """
    Class for Linear Reward Model.
    """

    def __init__(self, input_shape, layer_dims, lr=0.1, weight_decay=0, debug=False):
        super().__init__(input_shape, lr, weight_decay, debug)
        self.hidden_activation = None
        self.reward_activation = None
        self.reward_activation_type = None
        self.hidden_activation_type = None
        self.layer_dims = layer_dims
        self.layers = []
        self.initialize()

    def init_const(self, value=1.):
        for layer in self.layers:
            if isinstance(layer, nn.modules.linear.Linear):
                torch.nn.init.constant_(layer.weight.data, value)
                layer.bias.data.fill_(0.)
        torch.nn.init.constant_(self.fc_reward.weight.data, value)
        self.fc_reward.bias.data.fill_(0)

    def init_uniform(self, lo=-1, hi=1):
        for layer in self.layers:
            if isinstance(layer, nn.modules.linear.Linear):
                torch.nn.init.uniform_(layer.weight.data, lo, hi)
                layer.bias.data.fill_(0.)
        torch.nn.init.uniform_(self.fc_reward.weight.data, lo, hi)
        self.fc_reward.bias.data.fill_(0)

    def setup_encoder_layers(self):
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.hidden_activation = self.relu
        self.set_reward_activation(self.reward_activation_type)
        in_features = read_input_1d(self.in_shape())
        for out_features in self.layer_dims:
            self.layers.append(nn.Linear(in_features, out_features, bias=True))
            if self.hidden_activation is not None:
                self.layers.append(self.hidden_activation)
            in_features = out_features
        self.layers = torch.nn.ModuleList(self.layers)
        self.fc_reward = nn.Linear(in_features, 1, bias=True)

    def set_hidden_activation(self, type=None):
        assert type in ["relu", "sigmoid", "tanh", None]
        self.hidden_activation_type = type
        if type == "relu":
            self.hidden_activation = self.relu
        elif type == "sigmoid":
            self.hidden_activation = self.sigmoid
        elif type == "tanh":
            self.hidden_activation = self.tanh
        else:
            self.hidden_activation = None

    def set_reward_activation(self, type=None):
        assert type in ["relu", "sigmoid", None]
        self.reward_activation_type = type
        if type == "relu":
            self.reward_activation = self.relu
        elif type == "sigmoid":
            self.reward_activation = self.sigmoid
        else:
            self.reward_activation = None

    def get_state_dict(self):
        return self.state_dict()

    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def bottleneck_grads(self):
        return self.fc_reward.weight.grad

    # @staticmethod
    # def prepare_input(x):
    #     x = torch.FloatTensor(x)
    #     return x

    def encode(self, x, debug=False):
        for layer in self.layers:
            x = layer(x)
        return x

    def reward(self, x, return_latent=False, debug=False):
        z = self.encode(x, debug=debug)
        r = self.fc_reward(z)
        if self.reward_activation_type == "sigmoid":
            r = -self.reward_activation(r)
        elif self.reward_activation_type == "relu":
            r = -self.reward_activation(r)
        else:  # None
            pass
        # r -= 0.1
        if return_latent:
            return r, z
        else:
            return r

    def __call__(self, *args, **kwargs):
        return self.reward(*args, **kwargs)
