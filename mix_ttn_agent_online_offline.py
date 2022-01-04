import numpy as np
import torch as T
from TTN_network import TTNNetwork
from replay_memory import ReplayBuffer
import torch
from torch.autograd import Variable
# from sklearn.linear_model import Ridge
from tc.utils.tiles3 import *
# from numba import jit
# import nvidia_smi


class TTNAgent_online_offline_mix(object):
    def __init__(self, gamma, nnet_params, other_params, input_dims=10, num_units_rep=128, dir=None, offline=False,
                 num_tiling=16, num_tile=4, method_sarsa='expected-sarsa', tilecoding=1, replace_target_cnt=1000,
                 target_separate=False, status="online"):
        # gamma, loss_features, beta1, beta2, eps_init, eps_final, num_actions, replay_memory_size, replay_init_size, pretrain_rep_steps, freeze_rep,batch_size, fqi_reg_type, nn_lr, reg_A, eps_decay_steps, update_freq, input_dims, num_units_rep,
        # env_name='cacher', chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.loss_features = nnet_params['loss_features']
        self.beta1 = nnet_params['beta1']
        self.beta2 = nnet_params['beta2']
        self.eps_init = nnet_params['eps_init']
        self.epsilon = nnet_params['eps_init']
        self.eps_final = nnet_params['eps_final']
        self.eps_decay_steps = other_params['eps_decay_steps']
        self.n_actions = nnet_params['num_actions']
        self.batch_size = nnet_params['batch_size']
        self.fqi_reg_type = nnet_params['fqi_reg_type']
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
        self.hash_num = (self.num_tiles ** self.input_dims) * self.num_tilings
        self.iht = IHT(self.hash_num)
        self.status = status
        self.FIRST_N_FRAMES = nnet_params['FIRST_N_FRAMES']
        self.replay_init_size = nnet_params['replay_init_size']
        self.SQUARED_GRAD_MOMENTUM = nnet_params['SQUARED_GRAD_MOMENTUM']
        self.MIN_SQUARED_GRAD = nnet_params['MIN_SQUARED_GRAD']
        self.in_channels = nnet_params['in_channels']

        if self.input_dims == 4:
            self.obs_limits = [[-1, 1.0, 2.0], [-1, 1.0, 2.0], [-1, 1.0, 2.0], [-1, 1.0, 2.0]]
        else:
            self.obs_limits = [[-1.2, 0.6, 1.8], [-0.07, 0.07, 0.14]]

        self.update_feature = False
        self.method = method_sarsa
        self.tilecoding = tilecoding
        self.replace_target_cnt = replace_target_cnt
        self.target_saprate = target_separate

        self.memory = ReplayBuffer(nnet_params['replay_memory_size'], self.in_channels, self.input_dims, nnet_params['num_actions'],
                                   self.offline, self.memory_load_direction)
        # self.memory = self.assign_memory(nnet_params['replay_memory_size'], nnet_params['num_actions'])
        # self.q_eval,self.features, self.pred_states

        self.q_eval = TTNNetwork(self.beta1, self.beta2, self.lr, self.SQUARED_GRAD_MOMENTUM, self.MIN_SQUARED_GRAD, self.n_actions,
                                 input_dims=self.input_dims,
                                 in_channels = self.in_channels,
                                 number_unit=self.number_unit,
                                 num_units_rep=self.num_units_rep)

        self.q_next = TTNNetwork(self.beta1, self.beta2, self.lr, self.SQUARED_GRAD_MOMENTUM, self.MIN_SQUARED_GRAD, self.n_actions,
                                 input_dims=self.input_dims,
                                 in_channels = self.in_channels,
                                 number_unit=self.number_unit,
                                 num_units_rep=self.num_units_rep)

        # self.q_next, self.features_next, self.pred_states_next
        # self.q_next = TTNNetwork(self.beta1, self.beta2, self.lr, self.n_actions,
        #                             input_dims=self.input_dims,
        #                             name=self.env_name+'_'+self.algo+'_q_next',
        #                             chkpt_dir=self.chkpt_dir, number_unit=128, num_units_rep=self.num_units_rep)

        if self.tilecoding:
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

    # @jit(target='cuda')
    # @jit
    def decrement_epsilon(self):
        # self.epsilon = self.epsilon
        self.epsilon = self.eps_final if self.learn_step_counter - self.replay_init_size >= self.FIRST_N_FRAMES \
            else ((self.eps_final - self.eps_init) / self.FIRST_N_FRAMES) * (self.learn_step_counter - self.replay_init_size) + self.eps_init
        # print(self.epsilon)
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

        if np.random.random() > epsilon:
            with torch.no_grad():
                # state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                q_pred, features, pred_states = self.q_eval.forward(observation)
                features_bias = T.cat((features, T.ones((features.shape[0], 1)).to(self.q_eval.device)), 1)
                # print(features_bias)

                self.lin_values = self.update_lin_value(features_bias)
                action = self.lin_values.argmax()

            # action = T.argmax(q_pred).item()
        else:
            action = np.random.choice(self.action_space)

        return action

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

        for mmm in range(1):

            # self.learn_step_counter += 1

            # self.q_eval.optimizer.zero_grad()
            self.q_eval.zero_grad()

            # self.replace_target_network()

            states, actions, rewards, states_, actions_, dones = self.sample_memory_nextaction()
            indices = np.arange(self.batch_size)

            # q_pred = self.q_eval.forward(states)[indices, actions]
            q_pred_all, features_all, pred_states_all = self.q_eval.forward(states)
            q_pred = q_pred_all[indices, actions]
            # features = features_all
            if self.min > T.min(features_all):
                self.min = T.min(features_all)
            if self.max < T.max(features_all):
                self.max = T.max(features_all)
            # print(self.min, self.max)

            # pred_states_all = pred_states_all.view(-1, self.n_actions, self.input_dims)
            # pred_states = pred_states_all[indices, actions,:]

            # q_next = self.q_next.forward(states_).max(dim=1)[0]

            # loss = 0
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

                # loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
                # loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

            # if self.loss_features == "reward":  # reward loss
            #     loss = self.q_eval.loss(rewards, q_pred).to(self.q_eval.device)

            if self.loss_features == "next_state":  # next state loss
                # loss = self.q_eval.loss(states_, pred_states.squeeze()).to(self.q_eval.device)

                # _ , _, pred_states_next = self.q_eval.forward(states)
                pred_states_next_re = pred_states_all.view(-1, self.n_actions, self.input_dims)[indices, actions, :]
                loss = self.q_eval.loss((states_), (pred_states_next_re.squeeze())).to(self.q_eval.device)
                # print(q_pred.data, q_target.data)
                loss.backward()
                self.q_eval.optimizer.step()
            #
            # loss.backward()
            # self.q_eval.optimizer.step()
            # do update for q_next()

        # for FQI:
        if (self.learn_step_counter + 2) % self.update_freq == 0:
            print("FQI")
            for rep in range(self.fqi_rep):
                # print("num rep:", self.fqi_rep)
                with torch.no_grad():
                    mem_index = self.memory.mem_cntr if self.memory.mem_cntr < self.memory.mem_size else self.memory.mem_size
                    # convert them to pytorch array
                    if self.data_length == self.memory.mem_size:
                        states_all = T.tensor(self.memory.state_memory[:mem_index, :]).to(self.q_eval.device)
                        actions_all = T.tensor(self.memory.action_memory[:mem_index]).to(self.q_eval.device)
                        rewards_all = T.tensor(self.memory.reward_memory[:mem_index]).to(self.q_eval.device)
                        states_all_ = T.tensor(self.memory.new_state_memory[:mem_index, :]).to(self.q_eval.device)
                        actions_all_ = T.tensor(self.memory.new_action_memory[:mem_index]).to(self.q_eval.device)
                        dones_all = T.tensor(self.memory.terminal_memory[:mem_index]).to(self.q_eval.device)
                    else:
                        mem_index = min(min(self.memory.mem_cntr, self.memory.mem_size), self.data_length)
                        # print("length:", mem_index)
                        # states_all, actions_all, rewards_all, states_all_, actions_all_, dones_all = self.memory.sample_buffer_nextaction_consequtive(
                        #     self.data_length)
                        states_all, actions_all, rewards_all, states_all_, actions_all_, dones_all = self.memory.sample_buffer_nextaction_consequtive_chunk(self.data_length)

                    self.states_all_ch = states_all
                    self.actions_all_ch = actions_all
                    self.rewards_all_ch = rewards_all
                    self.states_all_ch_ = states_all_
                    self.actions_all_ch_ = actions_all_
                    self.dones_all_ch = dones_all


                    states_all_ch = states_all
                    actions_all_ch = actions_all
                    rewards_all_ch = rewards_all
                    states_all_ch_ = states_all_
                    # actions_all_ch_ = actions_all_
                    dones_all_ch = dones_all

                    # self.lin_values = T.mm(features, self.lin_weights)

                    q_next_allmem, features_nextmem, pred_states_nextmem = self.q_eval.forward(states_all_ch_)
                    features_nextmem_bias = T.cat(
                        (features_nextmem, T.ones((features_nextmem.shape[0], 1)).to(self.q_eval.device)), 1)
                    self.lin_values_next = self.update_lin_value(features_nextmem_bias)
                    maxlinq = T.max(self.lin_values_next, dim=1)[0].data
                    maxlinq[dones_all_ch] = 0
                    expectedsarsa = (1 - self.epsilon) * maxlinq + T.sum(
                        ((self.epsilon / self.n_actions) * self.lin_values_next.data), dim=1)
                    expectedsarsa[dones_all_ch] = 0

                    targets = rewards_all_ch + self.gamma * maxlinq
                    # targets = rewards_all_ch + self.gamma * expectedsarsa

                    _, features_allmem, _ = self.q_eval.forward(states_all_ch)
                    features_allmem_bias = T.cat(
                        (features_allmem, T.ones((features_allmem.shape[0], 1)).to(self.q_eval.device)), 1)

                    feats_current1 = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                                             features_allmem_bias.shape[1]).to(self.q_eval.device)

                    # for i in range(features_allmem_bias.shape[0]):
                    #     feats_current[i, actions_all_ch[i], :] = features_allmem_bias[i, :]

                    features_allmem_bias_re = T.reshape(features_allmem_bias, (
                    features_allmem_bias.shape[0], 1, features_allmem_bias.shape[1]))
                    actions_all_re1 = T.reshape(actions_all_ch, (actions_all_ch.shape[0], 1, 1))
                    actions_all_re = T.repeat_interleave(actions_all_re1, features_allmem_bias.shape[1], dim=2)
                    feats_current = feats_current1.scatter_(1, actions_all_re, features_allmem_bias_re)

                    feats_current = feats_current.view(-1, self.n_actions * features_allmem_bias.shape[1])

                    n = feats_current.shape[0]

                    if self.fqi_reg_type == 'prev':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        A = T.mm(A_tr, feats_current) + self.reg_A * T.eye(A_tr.shape[0])
                        b = T.mm(A_tr, T.unsqueeze(targets, 1)) + self.reg_A * self.lin_weights.reshape(-1, 1)
                        # b = T.mm(A_tr, T.unsqueeze(targets, 1)) + self.reg_A * T.reshape(T.transpose(self.lin_weights.reshape(-1, 1), 0, 1), [-1,1])
                        # print(A_tr.shape, A.shape, b.shape, targets.shape)
                        # new_weights = T.solve(b, A)[0]
                        # new_weights = T.lstsq(b, A)[0]  # T.mm(A.inverse(), b) #T.lstsq(b, A)[0]  #T.mm(A.inverse(), b) #tf.matrix_solve(A, b)
                        new_weights = T.linalg.lstsq(A, b).solution

                        #########################################

                    elif self.fqi_reg_type == 'l2':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        A = T.mm(A_tr, feats_current).to(self.q_eval.device)
                        b = T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.q_eval.device)
                        A += 1 * self.reg_A * T.eye(A.shape[0], A.shape[1]).to(self.q_eval.device)
                        # b += self.reg_A * self.lin_weights.reshape(-1, 1).to(self.q_eval.device)
                        new_weights = T.lstsq(b, A)[0].to(self.q_eval.device)
                    else:
                        raise AssertionError('fqi_reg_type is wrong')

                    # new_weights = T.transpose(T.reshape(T.squeeze(new_weights), [self.lin_weights.shape[1], self.lin_weights.shape[0]]), 0, 1)
                    # self.lin_weights.data = (new_weights.data)
                    # update_weights = new_weights
                    # self.lin_weights = T.transpose(new_weights.reshape(self.lin_weights.shape[1], self.lin_weights.shape[0]), 0, 1)

                    # self.lin_weights = new_weights.reshape(self.lin_weights.shape[0], self.lin_weights.shape[1])
                    # convex combination (Polyak-Ruppert Averaging)
                    self.lin_weights = self.tau * self.lin_weights + (1 - self.tau) * new_weights.reshape(
                        self.lin_weights.shape[0], self.lin_weights.shape[1])

        self.learn_step_counter += 1
        self.decrement_epsilon()

        return loss


    def learn_nn_feature(self, itr, shuffle_index):
        print("learn features with NN")
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
        print("learn features with NN")
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

        # for FQI:
        if (self.learn_step_counter + 2) % self.update_freq == 0:
            print("FQI")
            for rep in range(self.fqi_rep):
                # print("num rep:", self.fqi_rep)
                with torch.no_grad():
                    mem_index = self.memory.mem_cntr if self.memory.mem_cntr < self.memory.mem_size else self.memory.mem_size
                    # convert them to pytorch array
                    if self.data_length == self.memory.mem_size:
                        states_all = T.tensor(self.memory.state_memory[:mem_index, :]).to(self.q_eval.device)
                        actions_all = T.tensor(self.memory.action_memory[:mem_index]).to(self.q_eval.device)
                        rewards_all = T.tensor(self.memory.reward_memory[:mem_index]).to(self.q_eval.device)
                        states_all_ = T.tensor(self.memory.new_state_memory[:mem_index, :]).to(self.q_eval.device)
                        actions_all_ = T.tensor(self.memory.new_action_memory[:mem_index]).to(self.q_eval.device)
                        dones_all = T.tensor(self.memory.terminal_memory[:mem_index]).to(self.q_eval.device)
                    else:
                        mem_index = min(min(self.memory.mem_cntr, self.memory.mem_size), self.data_length)
                        # print("length:", mem_index)
                        # states_all, actions_all, rewards_all, states_all_, actions_all_, dones_all = self.memory.sample_buffer_nextaction_consequtive(
                        #     self.data_length)
                        states_all, actions_all, rewards_all, states_all_, actions_all_, dones_all = self.memory.sample_buffer_nextaction_consequtive_chunk(
                            self.data_length)

                    self.states_all_ch = states_all
                    self.actions_all_ch = actions_all
                    self.rewards_all_ch = rewards_all
                    self.states_all_ch_ = states_all_
                    self.actions_all_ch_ = actions_all_
                    self.dones_all_ch = dones_all

                    states_all_ch = states_all
                    actions_all_ch = actions_all
                    rewards_all_ch = rewards_all
                    states_all_ch_ = states_all_
                    # actions_all_ch_ = actions_all_
                    dones_all_ch = dones_all

                    # self.lin_values = T.mm(features, self.lin_weights)

                    q_next_allmem, features_nextmem, pred_states_nextmem = self.q_eval.forward(states_all_ch_)
                    features_nextmem_bias = T.cat(
                        (features_nextmem, T.ones((features_nextmem.shape[0], 1)).to(self.q_eval.device)), 1)
                    self.lin_values_next = self.update_lin_value(features_nextmem_bias)
                    maxlinq = T.max(self.lin_values_next, dim=1)[0].data
                    maxlinq[dones_all_ch] = 0
                    expectedsarsa = (1 - self.epsilon) * maxlinq + T.sum(
                        ((self.epsilon / self.n_actions) * self.lin_values_next.data), dim=1)
                    expectedsarsa[dones_all_ch] = 0

                    targets = rewards_all_ch + self.gamma * maxlinq
                    # targets = rewards_all_ch + self.gamma * expectedsarsa

                    _, features_allmem, _ = self.q_eval.forward(states_all_ch)
                    features_allmem_bias = T.cat(
                        (features_allmem, T.ones((features_allmem.shape[0], 1)).to(self.q_eval.device)), 1)

                    feats_current1 = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                                             features_allmem_bias.shape[1]).to(self.q_eval.device)

                    # for i in range(features_allmem_bias.shape[0]):
                    #     feats_current[i, actions_all_ch[i], :] = features_allmem_bias[i, :]

                    features_allmem_bias_re = T.reshape(features_allmem_bias, (
                        features_allmem_bias.shape[0], 1, features_allmem_bias.shape[1]))
                    actions_all_re1 = T.reshape(actions_all_ch, (actions_all_ch.shape[0], 1, 1))
                    actions_all_re = T.repeat_interleave(actions_all_re1, features_allmem_bias.shape[1], dim=2)
                    feats_current = feats_current1.scatter_(1, actions_all_re, features_allmem_bias_re)

                    feats_current = feats_current.view(-1, self.n_actions * features_allmem_bias.shape[1])

                    n = feats_current.shape[0]

                    if self.fqi_reg_type == 'prev':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        A = T.mm(A_tr, feats_current) + self.reg_A * T.eye(A_tr.shape[0])
                        b = T.mm(A_tr, T.unsqueeze(targets, 1)) + self.reg_A * self.lin_weights.reshape(-1, 1)
                        # b = T.mm(A_tr, T.unsqueeze(targets, 1)) + self.reg_A * T.reshape(T.transpose(self.lin_weights.reshape(-1, 1), 0, 1), [-1,1])
                        # print(A_tr.shape, A.shape, b.shape, targets.shape)
                        # new_weights = T.solve(b, A)[0]
                        new_weights = T.lstsq(b, A)[
                            0]  # T.mm(A.inverse(), b) #T.lstsq(b, A)[0]  #T.mm(A.inverse(), b) #tf.matrix_solve(A, b)

                        #########################################
                        # print("#############################")
                        # print('Using device:', self.q_eval.device)
                        # print("T.cuda.is_available():", T.cuda.is_available())
                        # print("T.cuda.current_device()", T.cuda.current_device())
                        # print("T.cuda.get_device_name(0)", T.cuda.get_device_name(0))
                        # print(T.cuda.device_count())
                        # print('Memory Usage:')
                        # print('Allocated:', round(T.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                        # print('Cached:   ', round(T.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

                        # # Additional Info when using cuda
                        # if self.q_eval.device.type == 'cuda':
                        #     print(T.cuda.get_device_name(0))
                        #     print('Memory Usage:')
                        #     print('Allocated:', round(T.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                        #     print('Cached:   ', round(T.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
                        # #
                        # nvidia_smi.nvmlInit()
                        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                        # # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
                        #
                        # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                        # print(" gpu:", res.gpu, "gpu-mem:", res.memory)

                        #########################################

                    elif self.fqi_reg_type == 'l2':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        A = T.mm(A_tr, feats_current).to(self.q_eval.device)
                        b = T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.q_eval.device)
                        A += 1 * self.reg_A * T.eye(A.shape[0], A.shape[1]).to(self.q_eval.device)
                        # b += self.reg_A * self.lin_weights.reshape(-1, 1).to(self.q_eval.device)
                        new_weights = T.lstsq(b, A)[0].to(self.q_eval.device)
                    else:
                        raise AssertionError('fqi_reg_type is wrong')

                    # new_weights = T.transpose(T.reshape(T.squeeze(new_weights), [self.lin_weights.shape[1], self.lin_weights.shape[0]]), 0, 1)
                    # self.lin_weights.data = (new_weights.data)
                    # update_weights = new_weights
                    # self.lin_weights = T.transpose(new_weights.reshape(self.lin_weights.shape[1], self.lin_weights.shape[0]), 0, 1)

                    # self.lin_weights = new_weights.reshape(self.lin_weights.shape[0], self.lin_weights.shape[1])
                    # convex combination (Polyak-Ruppert Averaging)
                    self.lin_weights = self.tau * self.lin_weights + (1 - self.tau) * new_weights.reshape(
                        self.lin_weights.shape[0], self.lin_weights.shape[1])

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

    def learn_fqi(self, feature, nextfeature):

        with torch.no_grad():

            L = self.states_all_ch.shape[0]

            # for FQI:
            # if (self.learn_step_counter + 2) % self.update_freq == 0 :
            for rep in range(self.fqi_rep):

                print("num rep:", self.fqi_rep)
                ctr = 0
                n = self.states_all_ch.shape[0]
                nsqrt = np.sqrt(self.states_all_ch.shape[0])
                A = 0
                b = 0
                for itr_mem in range(int(len(self.states_all_ch) / L)):

                    self.feature = feature[ctr * L: ctr * L + L]  # .to(self.q_eval.device)
                    self.nextfeature = nextfeature[ctr * L: ctr * L + L]  # .to(self.q_eval.device)

                    features_nextmem = self.nextfeature

                    # q_next_allmem, features_nextmem, pred_states_nextmem = self.q_eval.forward(self.states_all_ch_)

                    features_nextmem_bias = T.cat(
                        (features_nextmem, T.ones((features_nextmem.shape[0], 1)).to(self.q_eval.device)), 1)
                    self.lin_values_next = self.update_lin_value(features_nextmem_bias)
                    maxlinq = T.max(self.lin_values_next, dim=1)[0].data
                    # maxlinq[self.dones_all_ch[ctr*L: ctr*L+L]] = 0
                    expectedsarsa = (1 - self.epsilon) * maxlinq + T.sum(
                        ((self.epsilon / self.n_actions) * self.lin_values_next.data), dim=1)

                    # actions = self.actions_all_ch_[ctr * L: ctr * L + L].to(self.q_eval.device)
                    # sarsa = T.zeros(L).to(self.q_eval.device)
                    # for i in range(L):
                    #     sarsa[i] = self.lin_values_next[i, actions[i]]

                    if self.method == 'q-learning':
                        # print("q-learning")
                        maxlinq[self.dones_all_ch[ctr * L: ctr * L + L]] = 0
                        targets = self.rewards_all_ch[ctr * L: ctr * L + L] + self.gamma * maxlinq
                    # elif self.method == 'sarsa':
                    #     # print("sarsa")
                    #     sarsa[self.dones_all_ch[ctr * L: ctr * L + L]] = 0
                    #     targets = self.rewards_all_ch[ctr * L: ctr * L + L] + self.gamma * sarsa
                    elif self.method == 'expected-sarsa':
                        # print("expected-sarsa")
                        expectedsarsa[self.dones_all_ch[ctr * L: ctr * L + L]] = 0
                        targets = self.rewards_all_ch[ctr * L: ctr * L + L] + self.gamma * expectedsarsa
                    else:
                        raise AssertionError('method for fqi is wrong!')

                    # _, features_allmem, _ = self.q_eval.forward(states_all)
                    features_allmem = self.feature
                    features_allmem_bias = T.cat(
                        (features_allmem, T.ones((features_allmem.shape[0], 1)).to(self.q_eval.device)), 1)

                    feats_current = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                                            features_allmem_bias.shape[1]).to(self.q_eval.device)

                    actions_all_itr = self.actions_all_ch[ctr * L: ctr * L + L]
                    for i in range(features_allmem_bias.shape[0]):
                        feats_current[i, actions_all_itr[i], :] = features_allmem_bias[i, :]

                    # feats_current1 = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                    #                          features_allmem_bias.shape[1]).to(self.q_eval.device)
                    # features_allmem_bias_re = T.reshape(features_allmem_bias, (features_allmem_bias.shape[0], 1, features_allmem_bias.shape[1]))
                    # actions_all_re1 = T.reshape(self.actions_all_ch, (self.actions_all_ch.shape[0], 1, 1))
                    # actions_all_re = T.repeat_interleave(actions_all_re1, features_allmem_bias.shape[1], dim=2)
                    # feats_current = feats_current1.scatter_(1, actions_all_re, features_allmem_bias_re)

                    feats_current = feats_current.view(-1, self.n_actions * features_allmem_bias.shape[1])

                    if self.fqi_reg_type == 'prev':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        A += T.mm(A_tr, feats_current).to(self.q_eval.device)
                        b += T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.q_eval.device)

                    elif self.fqi_reg_type == 'l2':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        # A += (feats_current /nsqrt ).to(self.q_eval.device)
                        # b += (T.unsqueeze(targets, 1) / nsqrt).to(self.q_eval.device)
                        A += T.mm(A_tr, feats_current).to(self.q_eval.device)
                        b += T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.q_eval.device)

                    else:
                        raise AssertionError('fqi_reg_type is wrong')

                    # print(ctr)
                    ctr += 1
                    # if ctr == int(len(self.states_all_ch) / L): self.update_feature = True

                if self.states_all_ch.shape[0] > ctr * L:

                    self.feature = feature[ctr * L:]  # .to(self.q_eval.device)
                    self.nextfeature = nextfeature[ctr * L:]  # .to(self.q_eval.device)

                    features_nextmem = self.nextfeature

                    # q_next_allmem, features_nextmem, pred_states_nextmem = self.q_eval.forward(self.states_all_ch_)

                    features_nextmem_bias = T.cat(
                        (features_nextmem, T.ones((features_nextmem.shape[0], 1)).to(self.q_eval.device)), 1)
                    self.lin_values_next = self.update_lin_value(features_nextmem_bias)
                    maxlinq = T.max(self.lin_values_next, dim=1)[0].data
                    # maxlinq[self.dones_all_ch[ctr*L: ]] = 0
                    expectedsarsa = (1 - self.epsilon) * maxlinq + T.sum(
                        ((self.epsilon / self.n_actions) * self.lin_values_next.data), dim=1)
                    expectedsarsa[self.dones_all_ch[ctr * L:]] = 0

                    # targets = self.rewards_all_ch + self.gamma * maxlinq
                    targets = self.rewards_all_ch[ctr * L:] + self.gamma * expectedsarsa

                    # _, features_allmem, _ = self.q_eval.forward(states_all)
                    features_allmem = self.feature
                    features_allmem_bias = T.cat(
                        (features_allmem, T.ones((features_allmem.shape[0], 1)).to(self.q_eval.device)), 1)

                    feats_current = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                                            features_allmem_bias.shape[1]).to(self.q_eval.device)

                    actions_all_itr = self.actions_all_ch[ctr * L:]
                    for i in range(features_allmem_bias.shape[0]):
                        feats_current[i, actions_all_itr[i], :] = features_allmem_bias[i, :]

                    # feats_current1 = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                    #                          features_allmem_bias.shape[1]).to(self.q_eval.device)
                    # features_allmem_bias_re = T.reshape(features_allmem_bias, (features_allmem_bias.shape[0], 1, features_allmem_bias.shape[1]))
                    # actions_all_re1 = T.reshape(self.actions_all_ch, (self.actions_all_ch.shape[0], 1, 1))
                    # actions_all_re = T.repeat_interleave(actions_all_re1, features_allmem_bias.shape[1], dim=2)
                    # feats_current = feats_current1.scatter_(1, actions_all_re, features_allmem_bias_re)

                    feats_current = feats_current.view(-1, self.n_actions * features_allmem_bias.shape[1])

                    if self.fqi_reg_type == 'prev':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        A += T.mm(A_tr, feats_current).to(self.q_eval.device)
                        b += T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.q_eval.device)

                    elif self.fqi_reg_type == 'l2':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.q_eval.device)
                        # A += (feats_current / nsqrt).to(self.q_eval.device)
                        # b += (T.unsqueeze(targets, 1)/ nsqrt).to(self.q_eval.device)
                        A += T.mm(A_tr, feats_current).to(self.q_eval.device)
                        b += T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.q_eval.device)
                    else:
                        raise AssertionError('fqi_reg_type is wrong')

                    # print(ctr)
                    ctr += 1

                if self.fqi_reg_type == 'prev':
                    A += self.reg_A * T.eye(A_tr.shape[0]).to(self.q_eval.device)
                    b += self.reg_A * self.lin_weights.reshape(-1, 1).to(self.q_eval.device)
                    new_weights = T.lstsq(b, A)[0].to(
                        self.q_eval.device)  # T.mm(A.inverse(), b) #T.lstsq(b, A)[0]  #T.mm(A.inverse(), b) #tf.matrix_solve(A, b)
                elif self.fqi_reg_type == 'l2':
                    A += 1 * self.reg_A * T.eye(A.shape[0], A.shape[1]).to(self.q_eval.device)
                    # b += self.reg_A * self.lin_weights.reshape(-1, 1).to(self.q_eval.device)
                    new_weights = T.lstsq(b, A)[0].to(self.q_eval.device)
                    # new_weights = scipy.linalg.lstsq(A, b)[0]
                else:
                    raise AssertionError('fqi_reg_type is wrong')
                # new_weights = T.transpose(T.reshape(T.squeeze(new_weights), [self.lin_weights.shape[1], self.lin_weights.shape[0]]), 0, 1)
                # self.lin_weights.data = (new_weights.data)
                # update_weights = new_weights
                # self.lin_weights = T.transpose(new_weights.reshape(self.lin_weights.shape[1], self.lin_weights.shape[0]), 0, 1)

                # self.lin_weights = new_weights.reshape(self.lin_weights.shape[0], self.lin_weights.shape[1])
                # convex combination (Polyak-Ruppert Averaging)
                # self.lin_weights = self.tau* self.lin_weights + (1-self.tau) * new_weights.reshape(self.lin_weights.shape[0], self.lin_weights.shape[1])
                self.lin_weights = 0 * self.lin_weights + (1) * new_weights.reshape(
                    self.lin_weights.shape[0], self.lin_weights.shape[1])
            # self.learn_step_counter += 1

        self.decrement_epsilon()

        return

    def learn_pretrain(self):

        if self.tilecoding:
            feature = self.f_current
            nextfeature = self.f_next
            self.learn_fqi(feature, nextfeature)

        else:
            _, feature, _ = self.q_eval.forward(self.states_all_ch)
            _, nextfeature, _ = self.q_eval.forward(self.states_all_ch_)
            self.learn_fqi(feature, nextfeature)


