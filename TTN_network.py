import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
# from numba import jit





class TTNNetwork(nn.Module):
    def __init__(self, beta1, beta2, lr, n_actions, input_dims, sparse_matrix, number_unit=128, num_units_rep=128):
        super(TTNNetwork, self).__init__()

        self.sparse_matrix = sparse_matrix
        self.input_dims = input_dims

        # #hard code for testing
        # self.input_dims = 128



        # self.fc1 = nn.Linear(input_dims, number_unit, bias=True)
        # self.fc2 = nn.Linear(number_unit, number_unit, bias=True)
        # self.fc3 = nn.Linear(number_unit, num_units_rep, bias=True) # the representation layer
        self.rep = nn.Sequential(
            
            nn.Linear(self.sparse_matrix.shape[1], number_unit, bias=True),  ## For adding sparse inputs, need a better way to deal with it
            nn.ReLU(),
            nn.Linear(number_unit, number_unit, bias=True),
            nn.ReLU(),
            nn.Linear(number_unit, num_units_rep, bias=True),
            nn.ReLU(),
        )
        self.fc4 = nn.Linear(num_units_rep, n_actions, bias=True) # the prediction layer

        ## Use this layer to predict original state from the random sparse input passed throug rep part
        self.fc5 = nn.Linear(num_units_rep, n_actions*self.sparse_matrix.shape[1], bias=True) # the state-prediction layer 



        # nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu", mode='fan_in')
        # nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu", mode='fan_in')
        # nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu", mode='fan_in')
        # nn.init.kaiming_normal_(self.fc4.weight, nonlinearity="relu", mode='fan_in')
        self.rep.apply(self.init_weights)
        # self.fc1.bias.data.fill_(0.0)
        # self.fc2.bias.data.fill_(0.0)
        # self.fc3.bias.data.fill_(0.0)
        self.fc4.bias.data.fill_(0.0)

        # nn.init.zeros_(self.fc1.bias)
        # nn.init.zeros_(self.fc2.bias)
        # nn.init.zeros_(self.fc3.bias)
        # nn.init.zeros_(self.fc4.bias)

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0.0, amsgrad=True)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()


        # self.device = T.cuda.set_device(T.device('cuda:0'))
        # self.device = T.cuda.set_device(T.device('cuda'))
        self.device = T.device('cpu') #'cuda:0' if T.cuda.is_available() else 'cpu'
        self.to(self.device)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.)

    # @jit(target='CUDA 0')
    # @jit(nopython=True)
    # @jit
    def forward(self, state):
        # x = state
        """
        Build a network that maps state -> value-predictions, features, pred_states.
        """
        # print(state)
        # state = Variable(T.from_numpy(state))

        # print("T.cuda.is_available():", T.cuda.is_available())
        # print("T.cuda.current_device()", T.cuda.current_device())
        # print("T.cuda.get_device_name(0)", T.cuda.get_device_name(0))
        # Tesla K80

        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.rep(state)  # + T.zeros(1, self.input_dims)  # do we need to add bias
        self.predictions = self.fc4(x)
        self.pred_states = self.fc5(x)
        return self.predictions, x, self.pred_states



        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # self.features = F.relu(self.fc3(x)) #+ T.zeros(1, self.input_dims)  # do we need to add bias
        # self.predictions = self.fc4(self.features)
        # self.pred_states = self.fc5(self.features)
        # return self.predictions, self.features, self.pred_states

    # @jit(target='cuda')
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    # @jit(target='cuda')
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class TTNNetworkMaze(nn.Module):
    def __init__(self, beta1, beta2, lr, n_actions, input_dims, num_units_rep, chkpt_dir='checkpoint_files/', file_name='ttn_network_maze'):
        super(TTNNetworkMaze, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, file_name)
        rep_output_dim = 400 # Precalculated

        self.representation = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(rep_output_dim, num_units_rep),
            nn.ReLU(),
        )

        self.value_func = nn.Sequential(
            nn.Linear(num_units_rep, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

        self.value_func.apply(self.init_weights)
        # print(fc_input_dims)

        # self.fc1 = nn.Linear(fc_input_dims, 32)
        # self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0, amsgrad=True)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.)

    def forward(self, state):
        representation = self.representation(state)
        self.predictions = self.value_func(representation)
        self.pred_states = None # TODO: I'm not sure this would be a wise choice here
        return self.predictions, representation, self.pred_states

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))



# class TTNNetwork_image(nn.Module):
#     def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
#         super(TTNNetwork_image, self).__init__()
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

#         self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

#         fc_input_dims = self.calculate_conv_output_dims(input_dims)

#         self.fc1 = nn.Linear(fc_input_dims, 512)
#         self.fc2 = nn.Linear(512, n_actions)

#         self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

#         self.loss = nn.MSELoss()
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def calculate_conv_output_dims(self, input_dims):
#         state = T.zeros(1, *input_dims)
#         dims = self.conv1(state)
#         dims = self.conv2(dims)
#         dims = self.conv3(dims)
#         return int(np.prod(dims.size()))

#     def forward(self, state):
#         conv1 = F.relu(self.conv1(state))
#         conv2 = F.relu(self.conv2(conv1))
#         conv3 = F.relu(self.conv3(conv2))
#         # conv3 shape is BS x n_filters x H x W
#         conv_state = conv3.view(conv3.size()[0], -1)
#         # conv_state shape is BS x (n_filters * H * W)
#         flat1 = F.relu(self.fc1(conv_state))
#         actions = self.fc2(flat1)

#         return actions

#     def save_checkpoint(self):
#         print('... saving checkpoint ...')
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         print('... loading checkpoint ...')
#         self.load_state_dict(T.load(self.checkpoint_file))
