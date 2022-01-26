import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numba import jit
from utils.lta import tile_activation # sparsity import

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, 
                if_sparsity=False, tile_max=20, tile_min=-20, 
                 bins=20, eta=2, pre_tiling_width=20,
                 layers_to_apply=0, number_unit = 128):

        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.input_dims = input_dims
        
        self.if_sparsity = int(if_sparsity)
        self.bins = bins
        self.layers_to_apply_sparsity = layers_to_apply
        if eta < 0: # controls the sparsity
            self.eta = 1.0 / self.bins
        else:
            self.eta = eta

        self.tile_min = tile_min
        self.tile_max = tile_max

        # construct network based on sparsity.
        if self.if_sparsity == 0:
            print(f"No Sparsity is added!")
            self.fc1 = nn.Linear(input_dims, number_unit, bias=True)
            self.fc2 = nn.Linear(number_unit, number_unit, bias=True)
            self.fc3 = nn.Linear(number_unit, number_unit, bias=True)  # the representation layer
            self.fc4 = nn.Linear(number_unit, n_actions, bias=True)  # the prediction layer
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc1":
            print(f"Sparsity is added on {self.layers_to_apply_sparsity}!")
            self.fc1 = nn.Linear(input_dims, pre_tiling_width, bias=True)
            self.fc2 = nn.Linear(pre_tiling_width * bins, number_unit, bias=True)
            self.fc3 = nn.Linear(number_unit, number_unit, bias=True)  # the representation layer
            self.fc4 = nn.Linear(number_unit, n_actions, bias=True)  # the prediction layer
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc2":
            print(f"Sparsity is added on {self.layers_to_apply_sparsity}!")
            self.fc1 = nn.Linear(input_dims, number_unit, bias=True)
            self.fc2 = nn.Linear(number_unit, pre_tiling_width, bias=True)
            self.fc3 = nn.Linear(pre_tiling_width * bins, number_unit, bias=True)  # the representation layer
            self.fc4 = nn.Linear(number_unit, n_actions, bias=True)  # the prediction layer
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc1+fc2":
            print(f"Sparsity is added on {self.layers_to_apply_sparsity}!")
            self.fc1 = nn.Linear(input_dims, pre_tiling_width, bias=True)
            self.fc2 = nn.Linear(pre_tiling_width * bins, pre_tiling_width, bias=True)
            self.fc3 = nn.Linear(pre_tiling_width * bins, number_unit, bias=True)  # the representation layer
            self.fc4 = nn.Linear(number_unit, n_actions, bias=True)  # the prediction layer
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc3":
            print(f"Sparsity is added on {self.layers_to_apply_sparsity}!")
            self.fc1 = nn.Linear(input_dims, number_unit, bias=True)
            self.fc2 = nn.Linear(number_unit, number_unit, bias=True)
            self.fc3 = nn.Linear(number_unit, pre_tiling_width, bias=True)  # the representation layer
            self.fc4 = nn.Linear(pre_tiling_width * bins, n_actions, bias=True)  # the prediction layer

        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.xavier_uniform(self.fc4.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        # self.device = T.cuda.set_device(T.device('cuda:0'))
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # @jit(target='cuda:0')
    def forward(self, state):
        # x = state
        """
        Build a network that maps state -> value-predictions, features, pred_states.
        """
        # print(state)
        # state = Variable(T.from_numpy(state))
        if self.if_sparsity == 0:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))  # + T.zeros(1, self.input_dims)  # do we need to add bias
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc1":
            x = self.fc1(state)
            x = tile_activation(x, self.tile_min, self.tile_max, self.bins, self.eta)  # ITA applied to fc1's output.
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))  # + T.zeros(1, self.input_dims)  # do we need to add bias
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc2":
            x = F.relu(self.fc1(state))
            x = self.fc2(x)
            x = tile_activation(x, self.tile_min, self.tile_max, self.bins, self.eta)  # ITA applied to fc1's output.
            x = F.relu(self.fc3(x))  # + T.zeros(1, self.input_dims)  # do we need to add bias
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc3":
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)  # + T.zeros(1, self.input_dims)  # do we need to add bias
            x = tile_activation(x, self.tile_min, self.tile_max, self.bins, self.eta)  # ITA applied to fc1's output.
        elif self.if_sparsity == 1 and self.layers_to_apply_sparsity == "fc1+fc2":
            x = self.fc1(state)
            x = tile_activation(x, self.tile_min, self.tile_max, self.bins, self.eta)  # ITA applied to fc1's output.
            x = self.fc2(x)
            x = tile_activation(x, self.tile_min, self.tile_max, self.bins, self.eta)  # ITA applied to fc1's output.
            x = F.relu(self.fc3(x))  # + T.zeros(1, self.input_dims)  # do we need to add bias

        self.predictions = self.fc4(x)

        return self.predictions

    # @jit(target='cuda')
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    # @jit(target='cuda')
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    #     super(DeepQNetwork, self).__init__()
    #     self.checkpoint_dir = chkpt_dir
    #     self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
    #
    #     self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
    #     self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    #     self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
    #
    #     fc_input_dims = self.calculate_conv_output_dims(input_dims)
    #
    #     self.fc1 = nn.Linear(fc_input_dims, 512)
    #     self.fc2 = nn.Linear(512, n_actions)
    #
    #     self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
    #
    #     self.loss = nn.MSELoss()
    #     self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    #     self.to(self.device)
    #
    # def calculate_conv_output_dims(self, input_dims):
    #     state = T.zeros(1, *input_dims)
    #     dims = self.conv1(state)
    #     dims = self.conv2(dims)
    #     dims = self.conv3(dims)
    #     return int(np.prod(dims.size()))
    #
    # def forward(self, state):
    #     conv1 = F.relu(self.conv1(state))
    #     conv2 = F.relu(self.conv2(conv1))
    #     conv3 = F.relu(self.conv3(conv2))
    #     # conv3 shape is BS x n_filters x H x W
    #     conv_state = conv3.view(conv3.size()[0], -1)
    #     # conv_state shape is BS x (n_filters * H * W)
    #     flat1 = F.relu(self.fc1(conv_state))
    #     actions = self.fc2(flat1)
    #
    #     return actions
    #
    # def save_checkpoint(self):
    #     print('... saving checkpoint ...')
    #     T.save(self.state_dict(), self.checkpoint_file)
    #
    # def load_checkpoint(self):
    #     print('... loading checkpoint ...')
    #     self.load_state_dict(T.load(self.checkpoint_file))
