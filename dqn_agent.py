import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer




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

        self.memory = ReplayBuffer(nnet_params['replay_memory_size'], self.in_channels, self.input_dims,  self.n_actions, self.offline, self.memory_load_direction)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, self.in_channels, number_unit=128, chkpt_dir=self.chkpt_dir,
                                    name=self.env_name + '_' + self.algo + '_q_eval')

        self.q_next = DeepQNetwork(self.lr, self.n_actions, self.in_channels, number_unit=128, chkpt_dir=self.chkpt_dir,
                                    name=self.env_name + '_' + self.algo + '_q_eval')

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

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
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



