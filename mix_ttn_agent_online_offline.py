import numpy as np
import torch as T
from TTN_network import TTNNetwork, TTNNetworkMaze
from replay_memory import ReplayBuffer
import torch
from torch.autograd import Variable
from utils.data_augs import *
# from sklearn.linear_model import Ridge
from tc.utils.tiles3 import *
import warnings

# from numba import jit
# import nvidia_smi


class TTNAgent_online_offline_mix(object):
    def __init__(self, gamma, nnet_params, other_params, sparse_matrix, input_dims=4, num_units_rep=128, dir=None, offline=False,
                 num_tiling=16, num_tile=4, method_sarsa='expected-sarsa', tilecoding=1, replace_target_cnt=1000,
                 target_separate=False, status="online", data_aug_prob=""):
        # gamma, loss_features, beta1, beta2, eps_init, eps_final, num_actions, replay_memory_size, replay_init_size, pretrain_rep_steps, freeze_rep,batch_size, fqi_reg_type, nn_lr, reg_A, eps_decay_steps, update_freq, input_dims, num_units_rep,
        # env_name='cacher', chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.loss_features = nnet_params['loss_features']
        self.beta1 = nnet_params['beta1']
        self.beta2 = nnet_params['beta2']
        self.eps_init = nnet_params['eps_init']
        self.epsilon = 0.01  # nnet_params['eps_init']
        self.eps_final = nnet_params['eps_final']
        self.eps_decay_steps = other_params['eps_decay_steps']
        self.n_actions = nnet_params['num_actions']
        self.batch_size = nnet_params['batch_size']
        self.fqi_reg_type = nnet_params['fqi_reg_type']

        ##### Data Augmentation Params #####
        self.ras_alpha = other_params['ras_alpha'] # The minimum value of uniform distribution used for scaling the states in random amplitude scaling (rsa) technique
        self.ras_beta = other_params['ras_beta'] # The maximum value of uniform distribution used for scaling the states in random amplitude scaling (rsa) technique
        self.data_aug_type = other_params['data_aug_type'] # The type of data augmentation that we are going to use: random_shift (for visual inputs), rsa (for others)
        self.data_aug_prob = other_params['data_aug_prob'] # The probability of data getting augmented
        self.data_aug_loc = other_params['data_aug_loc'] # Where to apply augmentation: in rep learning, in fqi updates, or in both rep learning and fqi updates
        self.data_aug_pad = other_params['random_shift_pad'] # The number of pixels that will be padded for random shift technique. 4 usually works fine for this so there is no need to tune it

        ##### Input Transformation Params #####
        self.trans_type = other_params["trans_type"]
        self.new_feat_dim = other_params["new_feat_dim"]
        self.sparse_density = other_params["sparse_density"]
        
        self.lr = other_params['nn_lr']
        self.reg_A = other_params['reg_A']
        self.data_length = other_params['data_length']
        self.global_step = 5.5
        self.update_freq = other_params['update_freq']
        self.fqi_rep = other_params['fqi_rep']
        self.learn_step_counter = 0
        self.input_dims = input_dims
        self.num_units_rep = num_units_rep
        self.action_space = [i for i in range(nnet_params['num_actions'])]
        # self.eps_min = eps_min
        # self.eps_dec = eps_dec
        # self.replace_target_cnt = replace
        self.memory_load_direction = dir
        self.offline = offline
        self.number_unit = 128
        self.tau = 0.1  # 0.9 #0.1 #0.9 #0.5
        self.max = -100
        self.min = 100
        self.num_tiles = num_tile
        self.num_tilings = num_tiling

        if isinstance(self.input_dims, int): 
            self.hash_num = (self.num_tiles ** self.input_dims) * self.num_tilings
            self.iht = IHT(self.hash_num)
        else:
            warnings.warn('Tile coding is not defined for visual inputs!')
            
        self.status = status
        if isinstance(self.input_dims, int): 
            if self.input_dims == 4:
                self.obs_limits = [[-1, 1.0, 2.0], [-1, 1.0, 2.0], [-1, 1.0, 2.0], [-1, 1.0, 2.0]]
            else:
                self.obs_limits = [[-1.2, 0.6, 1.8], [-0.07, 0.07, 0.14]]
        else:
            warnings.warn('Tile coding is not defined for visual inputs!')

        self.update_feature = False
        self.method = method_sarsa
        self.tilecoding = tilecoding
        self.replace_target_cnt = replace_target_cnt
        self.target_saprate = target_separate

        self.memory = ReplayBuffer(nnet_params['replay_memory_size'], input_dims, nnet_params['num_actions'],
                                   self.offline, self.memory_load_direction)

        # self.memory = self.assign_memory(nnet_params['replay_memory_size'], nnet_params['num_actions'])
        # self.q_eval,self.features, self.pred_states
        if isinstance(self.input_dims, int): 
            self.q_eval = TTNNetwork(self.beta1, self.beta2, self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    number_unit=self.number_unit,
                                    num_units_rep=self.num_units_rep, sparse_matrix=sparse_matrix)

            self.q_next = TTNNetwork(self.beta1, self.beta2, self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    number_unit=self.number_unit,
                                    num_units_rep=self.num_units_rep, sparse_matrix=sparse_matrix)
        elif len(self.input_dims) == 3: #TODO: this part needs to be just for maze so maybe we have to add the environment as part of this class input. This part should be modified for minatar
            self.q_eval = TTNNetworkMaze(self.beta1, self.beta2, self.lr, self.n_actions,
                                    input_dims=self.input_dims, num_units_rep=self.num_units_rep)

            self.q_next = TTNNetworkMaze(self.beta1, self.beta2, self.lr, self.n_actions,
                                    input_dims=self.input_dims, num_units_rep=self.num_units_rep)            
        else:
            raise ValueError('Tile coding is not defined for this specifc input shape: {}'.format(self.input_dims))


        ##### Sparse Input Params ##### 
        self.sparse_matrix = sparse_matrix  ## What if no sparse inputs? how to define base sparse matrix which when applied does nothing.
        ## converting sparse matrix to tensor
        self.sparse_matrix = T.tensor(self.sparse_matrix, dtype=T.float).to(self.q_eval.device)

        
        # self.q_next, self.features_next, self.pred_states_next
        # self.q_next = TTNNetwork(self.beta1, self.beta2, self.lr, self.n_actions,
        #                             input_dims=self.input_dims,
        #                             name=self.env_name+'_'+self.algo+'_q_next',
        #                             chkpt_dir=self.chkpt_dir, number_unit=128, num_units_rep=self.num_units_rep)

        if self.tilecoding:
            if not isinstance(self.input_dims, int):
                raise ValueError('Tile coding cannot be used for visual inputs.')
            self.num_fe = self.hash_num + 1
        else:
            self.num_fe = self.num_units_rep + 1

        self.lin_weights = Variable(T.zeros(self.n_actions, (self.num_fe)))  # requires_grad=True
        self.lin_values = T.mm(Variable(T.zeros(self.batch_size, (self.num_fe))),
                               T.transpose(self.lin_weights, 0, 1))
        # self.clf = Ridge(alpha=self.reg_A)

        self.feature_list = []
        self.nextfeature_list = []

    # @jit(target='cuda')
    # @jit
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    # @jit(target='cuda')
    # @jit
    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    # @jit(target='cuda')
    # @jit
    def sample_memory_nextaction(self):
        state, action, reward, new_state, new_action, done = \
            self.memory.sample_buffer_nextaction(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        actions_ = T.tensor(new_action).to(self.q_eval.device)

        return states, actions, rewards, states_, actions_, dones


    def sample_memory_nextaction_shuffling(self, itr, shuffle_index):
        state, action, reward, new_state, new_action, done = \
                                self.memory.sample_buffer_nextaction_givenindex( self.batch_size, itr, shuffle_index)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        actions_ = T.tensor(new_action).to(self.q_eval.device)

        return states, actions, rewards, states_, actions_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    
    def sample_memory_nextaction_consequtive(self, sequence_size):
        state, action, reward, new_state, new_action, done = self.memory.sample_buffer_nextaction_consequtive(sequence_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        actions_ = T.tensor(new_action).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
    
    # @jit(target='cuda')
    # @jit
    def decrement_epsilon(self):
        
        #TODO: if TTN use fix , if DQN use decrement
        self.epsilon = self.epsilon


        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        # self.epsilon = self.eps_init * (self.eps_final ** (self.global_step / self.eps_decay_steps)) if self.epsilon > self.eps_final else self.eps_final

    # @jit(target='cuda')
    # @jit
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    # @jit(target='cuda')
    # @jit
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    # @jit(target='cuda')
    # @jit
    def choose_action(self, observation):
        if self.status == "offline":
            epsilon = self.epsilon
        elif self.status == "online":
            epsilon = self.epsilon
        elif self.status == "offline_online":
            epsilon = self.epsilon

        with torch.no_grad():
            #TODO: Do we need augmentation before choosing action?

            ##### Upscaling the input state with sparse matrix ##### 
            # state = T.matmul(state, self.sparse_matrix)

            # q_pred, _, _ = self.q_eval.forward(state)
            # features_bias = T.cat((features, T.ones((features.shape[0], 1)).to(self.q_eval.device)), 1)
            # # print(features_bias)

            # self.lin_values = self.update_lin_value(features_bias)
            # action = self.lin_values.argmax()

            # Q learning 
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            q_pred, _, _ = self.q_eval.forward(state)
            q_pred.detach()

        if np.random.random() > epsilon:
            action = T.argmax(q_pred).item()
        else:
            action = np.random.choice(self.action_space)

        return action, q_pred

    # @jit(target='cuda')
    # @jit
    def update_lin_value(self, features):
        # print(features)
        lin_values = T.mm(features, T.transpose(self.lin_weights, 0, 1))
        return lin_values

    # @jit(target='cuda:0')
    # @jit
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return


        #Representation learning part, i.e., Q-value learning
        for mmm in range(1):
            self.q_eval.zero_grad()

            states, actions, rewards, states_, actions_, dones = self.sample_memory_nextaction()
            indices = np.arange(self.batch_size)


            ##### Upscaling the input state with sparse matrix ##### 
            states = T.matmul(states, self.sparse_matrix)
            states_ = T.matmul(states_, self.sparse_matrix)


            #### Data Augmentation #####
            if (self.data_aug_prob > 0.0 and (self.data_aug_loc == 'rep' or self.data_aug_loc == 'both')):
                # print('augmenting data')
                if self.data_aug_type == 'random_shift':

                    states = random_shift(states, pad=self.random_shift_pad, p=self.data_aug_prob)
                    states_ = random_shift(states_, pad=self.random_shift_pad, p=self.data_aug_prob) 

                elif self.data_aug_type == 'ras':

                    states = random_amplitude_scaling(states, alpha=self.ras_alpha, beta=self.ras_beta, 
                                                        prob=self.data_aug_prob, multivariate=False)

                    states_ = random_amplitude_scaling(states_, alpha=self.ras_alpha, beta=self.ras_beta, 
                                                        prob=self.data_aug_prob, multivariate=False) 
                else:
                    raise ValueError('Data Augmentation type is not valid: ', self.data_aug_type)                                                       

            # q_pred = self.q_eval.forward(states)[indices, actions]
            q_pred_all, features_all, pred_states_all = self.q_eval.forward(states)
            q_pred = q_pred_all[indices, actions]
            # features = features_all
            if self.min > T.min(features_all):
                self.min = T.min(features_all)
            if self.max < T.max(features_all):
                self.max = T.max(features_all)
            # print(self.min, self.max)


            #TODO: Use target network here to make consistent with offline.
            if self.loss_features == "semi_MSTDE":
                with torch.no_grad():
                    q_next_all, features_next, pred_states_next = self.q_eval.forward(states_)
                    # q_next = q_next_all.max(dim=1)[0]
                    q_next = q_next_all[indices, actions_]
                    # print(q_next[dones])
                    q_next[dones] = 0.0

                    q_target = rewards + self.gamma * q_next

                loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
                # print(q_pred.data, q_target.data)
                loss.backward()
                self.q_eval.optimizer.step()


        # for Q-learning update:
        if (self.learn_step_counter + 2) % self.update_freq == 0:
            print("nn. learn Q-learning update: ", self.learn_step_counter)
            for rep in range(self.fqi_rep):

                self.q_eval.zero_grad()
                # Add loop to update on all states stored or load all data as in FQI updates
                # states, actions, rewards, states_, dones = self.sample_memory()
                states, actions, rewards, states_, dones = self.sample_memory_nextaction_consequtive(self.memory.mem_size)
                # print(f"states.shape: {states.shape} , states_.shape: {states_.shape}")
                # indices = np.arange(self.batch_size)
                indices = np.arange(self.memory.mem_size)
                q_pred, _, _ = self.q_eval.forward(states)
                q_pred = q_pred[indices, actions]

                with torch.no_grad():
                    if self.target_saprate:
                        print(f"Using target net: {self.target_saprate}")
                        self.replace_target_network()
                        q_next, _, _ = self.q_next.forward(states_)
                        q_next = q_next.max(dim=1)[0]
                    else:
                        q_next, _, _ = self.q_eval.forward(states_)
                        q_next = q_next.max(dim=1)[0]

                    q_next[dones] = 0.0
                    q_target = rewards + self.gamma*q_next

                loss_q = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
                loss_q.backward()
                self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

        return loss

        

    def learn_nn_feature(self, itr, shuffle_index):
        # print("learn features with NN")
        if self.memory.mem_cntr < self.batch_size:
            return

        self.learn_step_counter += 1

        # self.q_eval.optimizer.zero_grad()
        self.q_eval.zero_grad()


        states, actions, rewards, states_, actions_, dones = self.sample_memory_nextaction_shuffling(itr, shuffle_index)
        indices = np.arange(self.batch_size)

        # q_pred = self.q_eval.forward(states)[indices, actions]
        q_pred_all, features_all, pred_states_all = self.q_eval.forward(states)
        q_pred = q_pred_all[indices, actions]

        if self.min > T.min(features_all):
            self.min = T.min(features_all)
        if self.max < T.max(features_all):
            self.max = T.max(features_all)
        # print(self.min, self.max)


        if self.loss_features == "semi_MSTDE":
            with torch.no_grad():
                if self.target_saprate:
                    self.replace_target_network()
                    q_next_all, features_next, pred_states_next = self.q_next.forward(states_)
                else:
                    q_next_all, features_next, pred_states_next = self.q_eval.forward(states_)
                # q_next = q_next_all.max(dim=1)[0]
                q_next = q_next_all[indices, actions_]
                # print(q_next[dones])
                q_next[dones] = 0.0

                q_target = rewards + self.gamma * q_next


            loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
            # print(q_pred.data, q_target.data)
            loss.backward()
            self.q_eval.optimizer.step()




            # loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            # loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

        # if self.loss_features == "reward":  # reward loss
        #     loss = self.q_eval.loss(rewards, q_pred).to(self.q_eval.device)

        if self.loss_features == "next_state":  # next state loss
            # loss = self.q_eval.loss(states_, pred_states.squeeze()).to(self.q_eval.device)

            # _ , _, pred_states_next = self.q_eval.forward(states)
            pred_states_next_re = pred_states_all.view(-1, self.n_actions, self.input_dims)[indices, actions, :]
            loss = self.q_eval.loss((states_),(pred_states_next_re.squeeze())).to(self.q_eval.device)
            # print(q_pred.data, q_target.data)
            loss.backward()
            self.q_eval.optimizer.step()
        #
        # loss.backward()
        # self.q_eval.optimizer.step()
        # do update for q_next()

        self.decrement_epsilon()

        return loss


    def learn_nn_feature_fqi(self, itr, shuffle_index):
        # print("learn features with NN")
        if self.memory.mem_cntr < self.batch_size:
            return

        ## Why have two step counters?
        # self.learn_step_counter += 1

        # self.q_eval.optimizer.zero_grad()
        self.q_eval.zero_grad()


        states, actions, rewards, states_, actions_, dones = self.sample_memory_nextaction_shuffling(itr, shuffle_index)
        indices = np.arange(self.batch_size)


        ##### Upscaling the input state with sparse matrix ##### 
        states = T.matmul(states, self.sparse_matrix)
        states_ = T.matmul(states_, self.sparse_matrix)


        #### Data Augmentation #####
        if (self.data_aug_prob > 0.0 and (self.data_aug_loc == 'rep' or self.data_aug_loc == 'both')):
            # print('augmenting data')
            if self.data_aug_type == 'random_shift':

                states = random_shift(states, pad=self.random_shift_pad, p=self.data_aug_prob)
                states_ = random_shift(states_, pad=self.random_shift_pad, p=self.data_aug_prob) 

            elif self.data_aug_type == 'ras':

                states = random_amplitude_scaling(states, alpha=self.ras_alpha, beta=self.ras_beta, 
                                                    prob=self.data_aug_prob, multivariate=False)

                states_ = random_amplitude_scaling(states_, alpha=self.ras_alpha, beta=self.ras_beta, 
                                                    prob=self.data_aug_prob, multivariate=False) 
            else:
                raise ValueError('Data Augmentation type is not valid: ', self.data_aug_type)

        # q_pred = self.q_eval.forward(states)[indices, actions]
        q_pred_all, features_all, pred_states_all = self.q_eval.forward(states)
        q_pred = q_pred_all[indices, actions]

        if self.min > T.min(features_all):
            self.min = T.min(features_all)
        if self.max < T.max(features_all):
            self.max = T.max(features_all)
        # print(self.min, self.max)


        if self.loss_features == "semi_MSTDE":
            with torch.no_grad():
                q_next_all, features_next, pred_states_next = self.q_eval.forward(states_)
                # q_next = q_next_all.max(dim=1)[0]
                q_next = q_next_all[indices, actions_]
                # print(q_next[dones])
                q_next[dones] = 0.0

                q_target = rewards + self.gamma * q_next


            loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
            # print(q_pred.data, q_target.data)
            loss.backward()
            self.q_eval.optimizer.step()

        # for Q-learning:
        if (self.learn_step_counter + 2) % self.update_freq == 0:
            print("nn.learn_nn_feature_fqi Q-learning update: ", self.learn_step_counter)
            for rep in range(self.fqi_rep):

                self.q_eval.zero_grad()
                # Add loop to update on all states stored or load all data as in FQI updates
                # states, actions, rewards, states_, dones = self.sample_memory()
                states, actions, rewards, states_, dones = self.sample_memory_nextaction_consequtive(self.memory.mem_size)
                # print(f"states.shape: {states.shape} , states_.shape: {states_.shape}")
                # indices = np.arange(self.batch_size)
                indices = np.arange(self.memory.mem_size)

                q_pred, _, _ = self.q_eval.forward(states)
                q_pred = q_pred[indices, actions]

                with torch.no_grad():
                    if self.target_saprate:
                        print(f"Using target net: {self.target_saprate}")
                        self.replace_target_network()
                        q_next, _, _ = self.q_next.forward(states_)
                        q_next = q_next.max(dim=1)[0]
                    else:
                        q_next, _, _ = self.q_eval.forward(states_)
                        q_next = q_next.max(dim=1)[0]

                    q_next[dones] = 0.0
                    q_target = rewards + self.gamma*q_next

                loss_q = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
                loss_q.backward()
                self.q_eval.optimizer.step()
                
        self.learn_step_counter += 1
        self.decrement_epsilon()

        return loss

    def load_data(self):
        with torch.no_grad():

            mem_index = self.memory.mem_cntr if self.memory.mem_cntr < self.memory.mem_size else self.memory.mem_size

            states_all = T.tensor(self.memory.state_memory[:, :]).to(self.q_eval.device)
            actions_all = T.tensor(self.memory.action_memory[:]).to(self.q_eval.device)
            rewards_all = T.tensor(self.memory.reward_memory[:]).to(self.q_eval.device)
            states_all_ = T.tensor(self.memory.new_state_memory[:, :]).to(self.q_eval.device)
            actions_all_ = T.tensor(self.memory.new_action_memory[:]).to(self.q_eval.device)
            dones_all = T.tensor(self.memory.terminal_memory[:]).to(self.q_eval.device)
            # ep_all = T.tensor(self.memory.episode_memory[:]).to(self.q_eval.device)

            print(self.method)

            if self.offline:
                sp = 0 #100000
                ep = mem_index #sp+ 100000
                self.states_all_ch =states_all[sp:ep, :]
                self.actions_all_ch = actions_all[sp:ep]
                self.rewards_all_ch = rewards_all[sp:ep]
                self.states_all_ch_ = states_all_[sp:ep, :]
                self.actions_all_ch_ = actions_all_[sp:ep]
                self.dones_all_ch = dones_all[sp:ep]
                # self.ep_all_ch = ep_all[sp:ep]
                # L = 10000
            else:
                self.states_all_ch = states_all[:mem_index, :]
                self.actions_all_ch = actions_all[:mem_index]
                self.rewards_all_ch = rewards_all[:mem_index]
                self.states_all_ch_ = states_all_[:mem_index, :]
                self.actions_all_ch_ = actions_all_[:mem_index]
                self.dones_all_ch = dones_all[:mem_index]
                # self.ep_all_ch = ep_all[:mem_index]
                C = 50000
                if self.states_all_ch.shape[0]> C :
                    L = int(self.states_all_ch.shape[0] / C)
                else:
                    L = self.states_all_ch.shape[0]


    def tilecoding_feature(self):

        if self.update_feature == False:
            self.f_current = T.tensor(self.get_features_sparse(self.states_all_ch),
                                      dtype=T.float)  # .to(self.q_eval.device)
            self.f_next = T.tensor(self.get_features_sparse(self.states_all_ch_),
                                   dtype=T.float)  # .to(self.q_eval.device)

            self.update_feature = True

    def get_features_sparse(self, current_state):
        # print("get tile-coded features")
        scaled_obs = []
        features = T.zeros(len(current_state), self.hash_num).to(self.q_eval.device)
        for i in range(self.input_dims):
            # self.scaled_obs[i] = current_state[i]*self.num_tiles
            scaled_obs.append(((current_state[:, i] - self.obs_limits[i][0]) / self.obs_limits[i][2]) * self.num_tiles)
            # scaled_obs.append(((current_state[:, i] - self.obs_limits[i][0]) / self.obs_limits[i][2]) )
        for i in range(len(current_state)):
            if self.input_dims==4:
                current_scaled_obs = [scaled_obs[0][i], scaled_obs[1][i], scaled_obs[2][i], scaled_obs[3][i]]
            elif self.input_dims==2:
                current_scaled_obs = [scaled_obs[0][i], scaled_obs[1][i]]
            tiles_feature = tiles(self.iht, self.num_tilings, (current_scaled_obs))
            features[i, tiles_feature] = 1
        full = self.iht.full
        # if full:
        # print("iht is full!")
        return features

    def learn_fqi(self):

        self.q_eval.zero_grad()
        # Add loop to update on all states stored or load all data as in FQI updates
        # states, actions, rewards, states_, dones = self.sample_memory()
        states, actions, rewards, states_, dones = self.sample_memory_nextaction_consequtive(self.memory.mem_size)
        # print(f"states.shape: {states.shape} , states_.shape: {states_.shape}")
        # indices = np.arange(self.batch_size)
        indices = np.arange(self.memory.mem_size)
        q_pred, _, _ = self.q_eval.forward(states)
        q_pred = q_pred[indices, actions]

        with torch.no_grad():
            if self.target_saprate:
                print(f"Using target net: {self.target_saprate}")
                self.replace_target_network()
                q_next, _, _ = self.q_next.forward(states_)
                q_next = q_next.max(dim=1)[0]
            else:
                q_next, _, _ = self.q_eval.forward(states_)
                q_next = q_next.max(dim=1)[0]

            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next

        loss_q = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss_q.backward()
        self.q_eval.optimizer.step()

        return

    def learn_pretrain(self):
        print("nn.learn_pretrain")
        # if self.tilecoding:
        #     feature = self.f_current
        #     nextfeature = self.f_next
        #     self.learn_fqi(feature, nextfeature)

        # else:

        ##### Upscaling the input state with sparse matrix ##### 
        self.states_all_ch = T.matmul(self.states_all_ch, self.sparse_matrix)
        self.states_all_ch_ = T.matmul(self.states_all_ch_, self.sparse_matrix)

        #### Data Augmentation #####
        if (self.data_aug_prob > 0.0 and (self.data_aug_loc == 'fqi' or self.data_aug_loc == 'both')):
            # print('augmenting data')
            if self.data_aug_type == 'random_shift':

                self.states_all_ch = random_shift(self.states_all_ch, pad=self.random_shift_pad, p=self.data_aug_prob)
                self.states_all_ch_ = random_shift(self.states_all_ch_, pad=self.random_shift_pad, p=self.data_aug_prob) 

            elif self.data_aug_type == 'ras':

                self.states_all_ch = random_amplitude_scaling(self.states_all_ch, alpha=self.ras_alpha, beta=self.ras_beta, 
                                                    prob=self.data_aug_prob, multivariate=False)

                self.states_all_ch_ = random_amplitude_scaling(self.states_all_ch_, alpha=self.ras_alpha, beta=self.ras_beta, 
                                                    prob=self.data_aug_prob, multivariate=False) 
            else:
                raise ValueError('Data Augmentation type is not valid: ', self.data_aug_type)

        # q_pred, _, _ = self.q_eval.forward(self.states_all_ch)
        # q_next, _, _ = self.q_eval.forward(self.states_all_ch_)
        self.learn_fqi()


