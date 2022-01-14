import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import time
import random, argparse, logging
from collections import namedtuple
from minatar import Environment


class TTNNetwork(nn.Module):
    def __init__(self, beta1, beta2, lr, SQUARED_GRAD_MOMENTUM, MIN_SQUARED_GRAD, n_actions, input_dims=10,
                 in_channels=4, number_unit=128, num_units_rep=128, chkpt_dir='Online', name='Online'):
        super(TTNNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1

        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(input_dims) * size_linear_unit(input_dims) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=number_unit)

        # Grouping fc_hidden and conv together as the representation function (this is needed for ATC)
        self.rep = nn.Sequential(
            self.conv,
            nn.ReLU(),
            nn.Flatten(),
            self.fc_hidden,
        )
        self.conv_out_size = number_unit
        ######################################

        # Output layer:
        self.predictions = nn.Linear(in_features=128, out_features=n_actions)
        self.pred_states = nn.Linear(in_features=128, out_features=n_actions)  # ---> deconv or transformer?

        ### otimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0,
                                    amsgrad=True)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, alpha=SQUARED_GRAD_MOMENTUM, centered=True,
                                       eps=MIN_SQUARED_GRAD)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.device = T.cuda.set_device(T.device('cuda:0'))
        # self.device = T.cuda.set_device(T.device('cuda'))
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # # Rectified output from the first conv layer
        # x = F.relu(self.conv(x))
        # # Rectified output from the final hidden layer
        # x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))

        x= self.rep(x)

        # Returns the output from the fully-connected linear layer
        self.predictions(x)
        self.pred_states(x)

        return self.predictions(x), x, self.pred_states(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

#
# class TTNNetwork(nn.Module):
#     def __init__(self, beta1, beta2, lr, n_actions, input_dims, number_unit=128, num_units_rep=128):
#         super(TTNNetwork, self).__init__()
#
#         self.input_dims = input_dims
#
#
#
#         self.fc1 = nn.Linear(input_dims, number_unit, bias=True)
#         self.fc2 = nn.Linear(number_unit, number_unit, bias=True)
#         self.fc3 = nn.Linear(number_unit, num_units_rep, bias=True) # the representation layer
#         self.fc4 = nn.Linear(num_units_rep, n_actions, bias=True) # the prediction layer
#         self.fc5 = nn.Linear(num_units_rep, n_actions*input_dims, bias=True) # the state-prediction layer
#
#
#         # nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu", mode='fan_in')
#         # nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu", mode='fan_in')
#         # nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu", mode='fan_in')
#         # nn.init.kaiming_normal_(self.fc4.weight, nonlinearity="relu", mode='fan_in')
#         self.fc1.bias.data.fill_(0.0)
#         self.fc2.bias.data.fill_(0.0)
#         self.fc3.bias.data.fill_(0.0)
#         self.fc4.bias.data.fill_(0.0)
#
#         # nn.init.zeros_(self.fc1.bias)
#         # nn.init.zeros_(self.fc2.bias)
#         # nn.init.zeros_(self.fc3.bias)
#         # nn.init.zeros_(self.fc4.bias)
#
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)
#         nn.init.xavier_uniform_(self.fc4.weight)
#         nn.init.xavier_uniform_(self.fc5.weight)
#
#         self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=True)
#         # self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
#
#         self.loss = nn.MSELoss()
#
#
#
#
#         # self.device = T.cuda.set_device(T.device('cuda:0'))
#         # self.device = T.cuda.set_device(T.device('cuda'))
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)
#
#     # @jit(target='CUDA 0')
#     # @jit(nopython=True)
#     # @jit
#     def forward(self, state):
#         # x = state
#         """
#         Build a network that maps state -> value-predictions, features, pred_states.
#         """
#         # print(state)
#         # state = Variable(T.from_numpy(state))
#
#         # print("T.cuda.is_available():", T.cuda.is_available())
#         # print("T.cuda.current_device()", T.cuda.current_device())
#         # print("T.cuda.get_device_name(0)", T.cuda.get_device_name(0))
#         # Tesla K80
#
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))  # + T.zeros(1, self.input_dims)  # do we need to add bias
#         self.predictions = self.fc4(x)
#         self.pred_states = self.fc5(x)
#         return self.predictions, x, self.pred_states
#
#
#
#         # x = F.relu(self.fc1(state))
#         # x = F.relu(self.fc2(x))
#         # self.features = F.relu(self.fc3(x)) #+ T.zeros(1, self.input_dims)  # do we need to add bias
#         # self.predictions = self.fc4(self.features)
#         # self.pred_states = self.fc5(self.features)
#         # return self.predictions, self.features, self.pred_states
#
#     # @jit(target='cuda')
#     def save_checkpoint(self):
#         print('... saving checkpoint ...')
#         T.save(self.state_dict(), self.checkpoint_file)
#
#     # @jit(target='cuda')
#     def load_checkpoint(self):
#         print('... loading checkpoint ...')
#         self.load_state_dict(T.load(self.checkpoint_file))


