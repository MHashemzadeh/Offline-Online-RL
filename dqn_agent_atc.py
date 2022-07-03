import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer
from utils.data_augs import *
from utils.torch_utils import *
from utils.ul_networks import *
import copy

IGNORE_INDEX = -100

class DQNAgent(object):
    def __init__(self, gamma, nnet_params, other_params, input_dims=10, dir=None, offline=False, status="online"):

        self.gamma = gamma
        self.eps_init = nnet_params['eps_init']
        self.epsilon = 0.01 #nnet_params['eps_init']
        self.eps_final = nnet_params['eps_final']
        self.eps_decay_steps = other_params['eps_decay_steps']
        self.n_actions = nnet_params['num_actions']
        self.batch_size = nnet_params['batch_size']
        self.lr = other_params['nn_lr']
        self.global_step = 5.5
        self.replace_target_cnt = nnet_params['update_target_net_steps']
        self.input_dims = input_dims
        self.algo = 'dgn'
        self.env_name = 'catcher'
        self.chkpt_dir = 'tmp/dqn'
        self.action_space = [i for i in range( self.n_actions)]
        self.learn_step_counter = 0
        self.memory_load_direction = dir
        self.offline = offline
        self.status = status
        self.in_channels = nnet_params['in_channels']
        self.input_dims = input_dims
        self.FIRST_N_FRAMES = nnet_params['FIRST_N_FRAMES']
        self.replay_init_size = nnet_params['replay_init_size']
        self.SQUARED_GRAD_MOMENTUM = nnet_params['SQUARED_GRAD_MOMENTUM']
        self.MIN_SQUARED_GRAD = nnet_params['MIN_SQUARED_GRAD']
        ###############################################################
        self.data_aug_prob = 0.1  # other_params['data_aug_prob']
        self.data_aug_type = 'random_shift'  # other_params['data_aug_type']
        self.random_shift_pad = 4  # other_params['random_shift_pad']
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.ul_data_aug_type = self.data_aug_type
        self.ul_random_shift_prob = self.data_aug_prob
        self.ul_random_shift_pad = self.random_shift_pad
        self.ul_delta = 3  # other_params['delta']
        self.ul_delta_T = 3  # other_params['delta']
        self.ul_target_update_tau = 0.01
        self.total_steps = self.learn_step_counter
        self.ul_target_update_interval = 1
        self.ul_weigh = 1.0
        self.ul_clip_grad_norm = 10  # other_params['ul_clip_grad_norm']
        self.ul_lr = 1e-4  # 6.25e-5 #1e-3
        self.ATC = 1
        #########################################################

        self.memory = ReplayBuffer(nnet_params['replay_memory_size'], self.in_channels, self.input_dims,  self.n_actions, self.offline, self.memory_load_direction)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, self.in_channels, number_unit=128, chkpt_dir=self.chkpt_dir,
                                    name=self.env_name + '_' + self.algo + '_q_eval')

        self.q_next = DeepQNetwork(self.lr, self.n_actions, self.in_channels, number_unit=128, chkpt_dir=self.chkpt_dir,
                                    name=self.env_name + '_' + self.algo + '_q_eval')

        ##########################################################
        self.ul_latent_size = 256  # 128
        self.ul_anchor_hidden_sizes = 512

        self.ul_encoder = UlEncoerModel(
            conv=self.q_eval, #self.q_eval.rep,  # is using conv body for the behaviour network or the target network?
            latent_size=self.ul_latent_size,
            # self.number_unit, # i think this value will make it work if not you can tune it
            conv_out_size=self.q_eval.conv_out_size,
            # this is actually representation size not the output of conv networks
            device=self.device
        )
        self.ul_target_encoder = copy.deepcopy(self.ul_encoder)
        # self.ul_contrast = ContrastModel(self.n_actions, latent_size=self.number_unit, anchor_hidden_sizes=128) # anchor_hidden_sizes shouldn't be None, but you can sweep over this value
        self.ul_contrast = ContrastModel(self.n_actions, latent_size=self.ul_latent_size,
                                         anchor_hidden_sizes=self.ul_anchor_hidden_sizes)  # anchor_hidden_sizes shouldn't be None, but you can sweep over this value

        self.ul_parameters = list(self.ul_encoder.parameters()) + list(self.ul_contrast.parameters())
        self.ul_optimizer = T.optim.Adam(self.ul_parameters,
                                         lr=self.ul_lr)  # a different learning rate should be set for this network
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # def choose_action(self, observation):
    #     state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
    #     actions = self.q_eval.forward(state)
    #     if np.random.random() > self.epsilon:
    #         action = T.argmax(actions).item()
    #         actions.detach().numpy()
    #     else:
    #         action = np.random.choice(self.action_space)
    #
    #     return action, actions.detach()

    # @jit(target='cuda')
    def choose_action(self, observation):
        if self.status == "offline":
            epsilon = 0
        elif self.status == "online":
            epsilon = self.epsilon
        elif self.status == "offline_online":
            epsilon = self.epsilon

        with T.no_grad():
            # state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(observation)
            actions.detach()
        if np.random.random() > epsilon:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action, actions

    # @jit(target='cuda')
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    # @jit(target='cuda')
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
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    # @jit(target='cuda')
    def decrement_epsilon(self):
        # self.epsilon = self.epsilon
        self.epsilon = self.eps_final if self.learn_step_counter - self.replay_init_size >= self.FIRST_N_FRAMES \
            else ((self.eps_final - self.eps_init) / self.FIRST_N_FRAMES) * (
                    self.learn_step_counter - self.replay_init_size) + self.eps_init
        # print(self.epsilon)
        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        # self.epsilon = self.eps_init * (self.eps_final ** (self.global_step / self.eps_decay_steps)) if self.epsilon > self.eps_final else self.eps_final

    # @jit(target='cuda')
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    # @jit(target='cuda')
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


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

    # @jit(target='cuda')
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # print('Using device:', self.q_eval.device)
        # print()

        # Additional Info when using cuda
        if self.q_eval.device.type == 'cuda':
            print(T.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(T.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(T.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        ## loss1: calculate MSE for values estimation the same as DQN
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()


        ## loss2: Aug + ATC
        ## first step: Aug
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





        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss


    def learn_nn_feature(self, itr, shuffle_index):
        if self.memory.mem_cntr < self.batch_size:
            return

        # print('Using device:', self.q_eval.device)
        # print()

        # Additional Info when using cuda
        if self.q_eval.device.type == 'cuda':
            print(T.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(T.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(T.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, actions_, dones = self.sample_memory_nextaction_shuffling(itr, shuffle_index)
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss








