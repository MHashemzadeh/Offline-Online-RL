import numpy as np
import torch as T
from TTN_network import TTNNetwork
from replay_memory import ReplayBuffer
import torch
from torch.autograd import Variable
from utils.data_augs import *
from utils.torch_utils import *
from utils.ul_networks import *
# from sklearn.linear_model import Ridge
from tc.utils.tiles3 import *
# from numba import jit
# import nvidia_smi
import copy

IGNORE_INDEX = -100  # Mask contrast samples across episode boundary.

class TTNAgent_online_offline_mix(object):
    def __init__(self, gamma, nnet_params, other_params, input_dims=10, num_units_rep=128, dir=None, offline=False,
                 num_tiling=16, num_tile=4, method_sarsa='expected-sarsa', tilecoding=1, replace_target_cnt=1000,
                 target_separate=False, status="online"):
        # gamma, loss_features, beta1, beta2, eps_init, eps_final, num_actions, replay_memory_size, replay_init_size, pretrain_rep_steps, freeze_rep,batch_size, fqi_reg_type, nn_lr, reg_A, eps_decay_steps, update_freq, input_dims, num_units_rep,
        # env_name='cacher', chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.loss_features = 'ATC' #nnet_params['loss_features']
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
        ###############################################################
        self.data_aug_prob = 0.1 #other_params['data_aug_prob']
        self.data_aug_type = 'random_shift' #other_params['data_aug_type']
        self.random_shift_pad = 4 #other_params['random_shift_pad']
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.ul_data_aug_type = self.data_aug_type
        self.ul_random_shift_prob = self.data_aug_prob
        self.ul_random_shift_pad = self.random_shift_pad
        self.ul_delta = 3 #other_params['delta']
        self.ul_delta_T = 3 #other_params['delta']
        self.ul_target_update_tau = 0.01
        self.total_steps = self.learn_step_counter
        self.ul_target_update_interval = 1
        self.ul_weigh = 1.0
        self.ul_clip_grad_norm = 10 #other_params['ul_clip_grad_norm']
        self.ul_lr = 1e-4 #6.25e-5 #1e-3
        self.ATC = 1



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

        ####################### ATC Loss Network Init #####################
        # batch_size_cl = 32  # should be the same
        # augmentation_padding = 4  # should be the same
        # augmentation_prob = 0.1  # [0.01, 0.1, 0.2, 0.3 , 1.] should be defined in the code
        # delta = 3  # should be the same
        # target_update_interval = 1
        # ul_target_update_tau = 0.01  # should be the same
        # ul_clip_grad_norm = 10.
        self.ul_latent_size=256 # 128
        self.ul_anchor_hidden_sizes=512

        self.ul_encoder = UlEncoerModel(
                                conv=self.q_eval.rep, # is using conv body for the behaviour network or the target network?
                                latent_size= self.ul_latent_size, #self.number_unit, # i think this value will make it work if not you can tune it
                                conv_out_size=self.q_eval.conv_out_size, # this is actually representation size not the output of conv networks
                                device=self.device
                            )
        self.ul_target_encoder = copy.deepcopy(self.ul_encoder)
        # self.ul_contrast = ContrastModel(self.n_actions, latent_size=self.number_unit, anchor_hidden_sizes=128) # anchor_hidden_sizes shouldn't be None, but you can sweep over this value
        self.ul_contrast = ContrastModel(self.n_actions, latent_size=self.ul_latent_size , anchor_hidden_sizes=self.ul_anchor_hidden_sizes)  # anchor_hidden_sizes shouldn't be None, but you can sweep over this value

        self.ul_parameters = list(self.ul_encoder.parameters()) + list(self.ul_contrast.parameters())
        self.ul_optimizer = T.optim.Adam(self.ul_parameters, lr=self.ul_lr) # a different learning rate should be set for this network
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


        self.feature_list = []
        self.nextfeature_list = []

    def ul_parameters(self):
        yield from self.ul_encoder.parameters()
        yield from self.ul_contrast.parameters()

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
                _, features, _ = self.q_eval.forward(observation)
                features_bias = T.cat((features, T.ones((features.shape[0], 1)).to(self.device)), 1)
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

            if self.data_aug_prob > 0.:
                # print('augmenting data')
                if self.data_aug_type == 'random_shift':

                    states = random_shift(states, pad=self.random_shift_pad, prob=self.data_aug_prob)
                    states_ = random_shift(states_, pad=self.random_shift_pad, prob=self.data_aug_prob)

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

            if self.ATC:  #if self.loss_features == "ATC": #if self.ATC:  #
                batch_size_cl = 32  # should be the same

                ########### Computing the contrastive loss ###########
                # Currently it is implemented only with a single environment in mind and one repetition
                # print('Calculating CL loss')
                self.ul_optimizer.zero_grad()
                states, actions, rewards, next_states, _, terminals = self.memory.sample_buffer_nextaction_consequtive(batch_size_cl)
                # print(terminals)
                anchor = states[:-self.ul_delta_T]
                positive = states[self.ul_delta_T:]

                # print('anchor: ', anchor.shape)
                # print('positive: ', positive.shape)
                # print(self.ul_random_shift_prob)
                # print(self.ul_random_shift_pad)
                if self.ul_random_shift_prob > 0.:

                    if self.ul_data_aug_type == 'random_shift': ## for image

                        anchor = random_shift(imgs=anchor, pad=self.ul_random_shift_pad, prob=self.ul_random_shift_prob)
                        positive = random_shift(imgs=positive, pad = self.ul_random_shift_pad, prob= self.ul_random_shift_prob)

                    elif self.ul_data_aug_type == 'ras':

                        anchor = random_amplitude_scaling(anchor, alpha=self.ul_ras_alpha, beta=self.ul_ras_beta,
                                                          prob=self.ul_random_shift_prob, multivariate=False)

                        positive = random_amplitude_scaling(positive, alpha=self.ul_ras_alpha, beta=self.ul_ras_beta,
                                                            prob=self.ul_random_shift_prob, multivariate=False)
                    else:
                        raise ValueError('Data Augmentation type is not valid: ', self.ul_data_aug_type)

                        # anchor = random_shift(
                    #     imgs=anchor,
                    #     pad=self.ul_random_shift_pad,
                    #     prob=self.ul_random_shift_prob,
                    # )

                    # positive = random_shift(
                    #     imgs=positive,
                    #     pad=self.ul_random_shift_pad,
                    #     prob=self.ul_random_shift_prob,
                    # )
                # anchor, positive = buffer_to((anchor, positive),
                #    device=self.agent.device)
                with torch.no_grad():
                    c_positive = self.ul_target_encoder(positive)
                c_anchor = self.ul_encoder(anchor)
                logits = self.ul_contrast(c_anchor, c_positive)  # anchor mlp in here.

                labels = torch.arange(c_anchor.shape[0],
                                      dtype=torch.long, device=self.device)
                terminals = torch_utils.tensor(terminals, self.device)
                valid = valid_from_done(terminals).type(torch.bool)  # use all
                valid = valid[self.ul_delta_T:].reshape(-1)  # at positions of positive
                labels[~valid] = IGNORE_INDEX

                ul_loss =  self.c_e_loss(logits, labels)
                ul_loss.backward()
                if self.ul_clip_grad_norm is None:
                    grad_norm = 0.
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.ul_parameters, self.ul_clip_grad_norm)
                self.ul_optimizer.step()

                # Just logging purposes
                correct = torch.argmax(logits.detach(), dim=1) == labels
                accuracy = torch.mean(correct[valid].float())

                if self.total_steps % self.ul_target_update_interval == 0:
                    update_state_dict(self.ul_target_encoder, self.ul_encoder.state_dict(),
                                      self.ul_target_update_tau)


        # for FQI:
        if (self.learn_step_counter + 2) % self.update_freq == 0:
            print("FQI")
            for rep in range(self.fqi_rep):
                # print("num rep:", self.fqi_rep)
                with torch.no_grad():
                    mem_index = self.memory.mem_cntr if self.memory.mem_cntr < self.memory.mem_size else self.memory.mem_size
                    # convert them to pytorch array
                    if self.data_length == self.memory.mem_size:
                        states_all = T.tensor(self.memory.state_memory[:mem_index, :]).to(self.device)
                        actions_all = T.tensor(self.memory.action_memory[:mem_index]).to(self.device)
                        rewards_all = T.tensor(self.memory.reward_memory[:mem_index]).to(self.device)
                        states_all_ = T.tensor(self.memory.new_state_memory[:mem_index, :]).to(self.device)
                        actions_all_ = T.tensor(self.memory.new_action_memory[:mem_index]).to(self.device)
                        dones_all = T.tensor(self.memory.terminal_memory[:mem_index]).to(self.device)
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

                    if self.data_aug_prob > 0.:
                        # print('augmenting data')
                        if self.data_aug_type == 'random_shift':

                            states_all_ch = random_shift(states_all_ch, pad=self.random_shift_pad, prob=self.data_aug_prob)
                            states_all_ch_ = random_shift(states_all_ch_, pad=self.random_shift_pad, prob=self.data_aug_prob)

                        elif self.data_aug_type == 'ras':

                            states_all_ch = random_amplitude_scaling(states_all_ch, alpha=self.ras_alpha, beta=self.ras_beta,
                                                              prob=self.data_aug_prob, multivariate=False)

                            states_all_ch_ = random_amplitude_scaling(states_all_ch_, alpha=self.ras_alpha, beta=self.ras_beta,
                                                               prob=self.data_aug_prob, multivariate=False)
                        else:
                            raise ValueError('Data Augmentation type is not valid: ', self.data_aug_type)



                    q_next_allmem, features_nextmem, pred_states_nextmem = self.q_eval.forward(states_all_ch_)
                    # features_nextmem = self.ul_encoder.reps(states_all_ch_)

                    features_nextmem_bias = T.cat(
                        (features_nextmem, T.ones((features_nextmem.shape[0], 1)).to(self.device)), 1)
                    self.lin_values_next = self.update_lin_value(features_nextmem_bias)
                    maxlinq = T.max(self.lin_values_next, dim=1)[0].data
                    maxlinq[dones_all_ch] = 0
                    expectedsarsa = (1 - self.epsilon) * maxlinq + T.sum(
                        ((self.epsilon / self.n_actions) * self.lin_values_next.data), dim=1)
                    expectedsarsa[dones_all_ch] = 0

                    targets = rewards_all_ch + self.gamma * maxlinq
                    # targets = rewards_all_ch + self.gamma * expectedsarsa

                    _, features_allmem, _ = self.q_eval.forward(states_all_ch)
                    # features_allmem = self.ul_encoder.reps(states_all_ch)

                    features_allmem_bias = T.cat(
                        (features_allmem, T.ones((features_allmem.shape[0], 1)).to(self.device)), 1)

                    feats_current1 = T.zeros(features_allmem_bias.shape[0], self.n_actions,
                                             features_allmem_bias.shape[1]).to(self.device)

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
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.device)
                        A = T.mm(A_tr, feats_current) + self.reg_A * T.eye(A_tr.shape[0])
                        b = T.mm(A_tr, T.unsqueeze(targets, 1)) + self.reg_A * self.lin_weights.reshape(-1, 1)
                        # b = T.mm(A_tr, T.unsqueeze(targets, 1)) + self.reg_A * T.reshape(T.transpose(self.lin_weights.reshape(-1, 1), 0, 1), [-1,1])
                        # print(A_tr.shape, A.shape, b.shape, targets.shape)
                        # new_weights = T.solve(b, A)[0]
                        # new_weights = T.lstsq(b, A)[0]  # T.mm(A.inverse(), b) #T.lstsq(b, A)[0]  #T.mm(A.inverse(), b) #tf.matrix_solve(A, b)
                        new_weights = T.linalg.lstsq(A, b).solution

                        #########################################

                    elif self.fqi_reg_type == 'l2':
                        A_tr = (T.transpose(feats_current, 0, 1) / n).to(self.device)
                        A = T.mm(A_tr, feats_current).to(self.device)
                        b = T.mm(A_tr, T.unsqueeze(targets, 1)).to(self.device)
                        A += 1 * self.reg_A * T.eye(A.shape[0], A.shape[1]).to(self.device)
                        # b += self.reg_A * self.lin_weights.reshape(-1, 1).to(self.device)
                        new_weights = T.lstsq(b, A)[0].to(self.device)
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

        return accuracy



