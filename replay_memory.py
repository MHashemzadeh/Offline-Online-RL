import numpy as np
import torch as T
# from numba import jit
np_load_old = np.load

# modify the default parameters of np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

class ReplayBuffer(object):
    def __init__(self, max_size, in_channels, input_dims, n_actions, offline=False, dir=None, uniform=False):
        self.mem_size = max_size
        self.dir = dir
        self.uniform = uniform
        self.input_dims = input_dims
        self.in_channels = in_channels

        if offline == False:
            self.state_memory = T.zeros((self.mem_size, self.in_channels, self.input_dims, self.input_dims),  dtype=T.float32)
            self.new_state_memory = T.zeros((self.mem_size, self.in_channels, self.input_dims, self.input_dims), dtype=T.float32)
            self.action_memory = T.zeros(self.mem_size, dtype=T.int64)
            self.new_action_memory = T.zeros(self.mem_size, dtype=T.int64)
            self.reward_memory = T.zeros(self.mem_size, dtype=T.float32)
            self.terminal_memory = T.zeros(self.mem_size, dtype=T.bool)
            self.mem_cntr = 0
        elif uniform == True:
            # load offline data
            print("load offline data Uniform!!!!!")
            d = np.load(self.dir)
            # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])
            self.state_memory = T.tensor((d.item().get('state')), dtype=T.float32)
            self.new_state_memory = T.tensor((d.item().get('nstate')), dtype=T.float32)
            self.action_memory = T.tensor((d.item().get('action')), dtype=T.int64)
            self.reward_memory = T.tensor((d.item().get('reward')), dtype=T.float32)
            self.terminal_memory = T.tensor((d.item().get('done')), dtype=T.bool)
            self.mem_cntr = self.mem_size
        else:
            print("load offline data!!!!!")
            d = np.load(self.dir)
            # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])
            self.state_memory = T.zeros((self.mem_size, self.in_channels, self.input_dims, self.input_dims), dtype=T.float32)
            self.new_state_memory = T.zeros((self.mem_size, self.in_channels, self.input_dims, self.input_dims), dtype=T.float32)
            self.action_memory = T.zeros(self.mem_size, dtype=T.int64)
            self.new_action_memory = T.zeros(self.mem_size, dtype=T.int64)
            self.reward_memory = T.zeros(self.mem_size, dtype=T.float32)
            self.terminal_memory = T.zeros(self.mem_size, dtype=T.bool)

            temp = T.tensor((d.item().get('state')), dtype=T.float32)

            self.state_memory[:len(temp), :] = T.tensor((d.item().get('state')), dtype=T.float32)
            self.new_state_memory[:len(temp), :] = T.tensor((d.item().get('nstate')), dtype=T.float32)
            self.action_memory[:len(temp)] = T.tensor((d.item().get('action')), dtype=T.int64)
            self.new_action_memory[:len(temp)] = T.tensor((d.item().get('naction')), dtype=T.int64)
            self.reward_memory[:len(temp)] = T.tensor((d.item().get('reward')), dtype=T.float32)
            self.terminal_memory[:len(temp)] = T.tensor((d.item().get('done')), dtype=T.bool)
            self.mem_cntr = len(temp)

            # self.state_memory = self.state_memory[:self.mem_size]
            # self.new_state_memory = self.new_state_memory[:self.mem_size]
            # self.action_memory = self.action_memory[:self.mem_size]
            # self.new_action_memory = self.new_action_memory[:self.mem_size]
            # self.reward_memory = self.reward_memory[:self.mem_size]
            # self.terminal_memory = self.terminal_memory[:self.mem_size]
            # self.mem_cntr = self.mem_size

    # @jit(target='cuda')
    # @jit
    def store_transition(self, state, action, reward, state_, done):
        # print("stor!!!!!!!!!!!!!!!!!!!!!!!!")
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = T.tensor(state)
        self.new_state_memory[index] = T.tensor(state_)
        self.action_memory[index] = T.tensor(action)
        self.reward_memory[index] = T.tensor(reward)
        self.terminal_memory[index] = T.tensor(done)
        self.mem_cntr += 1

    # @jit(target='cuda')
    # @jit
    def store_transition_withnewaction(self, state, action, reward, state_, action_, done):
        # print("stor!!!!!!!!!!!!!!!!!!!!!!!!")
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = T.tensor(state)
        self.new_state_memory[index] = T.tensor(state_)
        self.action_memory[index] = T.tensor(action)
        self.new_action_memory[index] = T.tensor(action_)
        self.reward_memory[index] = T.tensor(reward)
        self.terminal_memory[index] = T.tensor(done)
        self.mem_cntr += 1

    # @jit(target='cuda')
    # @jit
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    # @jit(target='cuda')
    # @jit
    def sample_buffer_nextaction(self, batch_size):
        max_mem= min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        actions_ = self.new_action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, actions_, terminal


    def sample_buffer_nextaction_givenindex(self, batch_size, itr, shuffle_index):
        start_ind = itr* batch_size
        batch = shuffle_index[start_ind: start_ind+batch_size]
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        actions_ = self.new_action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, actions_, terminal

    # @jit(target='cuda')
    # @jit
    def sample_buffer_nextaction_consequtive(self, sequence_size):
        max_mem= min(self.mem_cntr, self.mem_size)
        endpoint = max( 1, (max_mem - sequence_size))
        startpoint = np.random.choice(endpoint, 1, replace=False)
        ## for the last chunk only
        # SP = endpoint
        # EP = min((endpoint+sequence_size), max_mem)
        ## for general case
        SP = np.int(startpoint)
        EP = np.int(min((SP + sequence_size), max_mem))
        states = self.state_memory[SP:EP]
        actions = self.action_memory[SP:EP]
        rewards = self.reward_memory[SP:EP]
        states_ = self.new_state_memory[SP:EP]
        actions_ = self.new_action_memory[SP:EP]
        terminal = self.terminal_memory[SP:EP]
        # states = self.state_memory[-max_mem:]
        # actions = self.action_memory[-max_mem:]
        # rewards = self.reward_memory[-max_mem:]
        # states_ = self.new_state_memory[-max_mem:]
        # actions_ = self.new_action_memory[-max_mem:]
        # terminal = self.terminal_memory[-max_mem:]

        return states, actions, rewards, states_, actions_, terminal

    # @jit(target='cuda')
    # @jit
    def sample_buffer_nextaction_consequtive_chunk(self, sequence_size, chunk_num=1):
        # print("chunk_num", chunk_num)
        # print("sequence_size:", sequence_size)
        max_mem= min(self.mem_cntr, self.mem_size)
        max_mem_per_chunk = np.int(max_mem / chunk_num)
        sequence_size_per_chunk = np.int(sequence_size / chunk_num)
        mem_st =0
        states = []
        actions = []
        rewards = []
        states_ = []
        actions_ = []
        terminal = []
        for i in range(chunk_num):
            mem = mem_st + max_mem_per_chunk
            endpoint = max( 1, (mem - sequence_size_per_chunk))
            startpoint = np.random.randint(mem_st, endpoint) #np.random.choice([mem_st, endpoint], 1, replace=False)
            ## for the last chunk only
            # SP = endpoint
            # EP = min((endpoint+sequence_size), max_mem)
            ## for general case
            SP = np.int(startpoint)
            EP = np.int(min((SP + sequence_size_per_chunk), mem))
            states.append(self.state_memory[SP:EP])
            actions.append(self.action_memory[SP:EP])
            rewards.append(self.reward_memory[SP:EP])
            states_.append(self.new_state_memory[SP:EP])
            actions_.append(self.new_action_memory[SP:EP])
            terminal.append(self.terminal_memory[SP:EP])
            mem_st = mem_st + max_mem_per_chunk

        states = T.cat(states, dim=0)
        actions = T.cat(actions, dim=0)
        rewards = T.cat(rewards, dim=0)
        states_ = T.cat(states_, dim=0)
        actions_ = T.cat(actions_, dim=0)
        terminal = T.cat(terminal, dim=0)

        return states, actions, rewards, states_, actions_, terminal

    def load_mem(self):

        if self.uniform == True:
            # load offline data
            print("load offline data Uniform!!!!!")
            d = np.load(self.dir)
            # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])
            self.state_memory = T.tensor((d.item().get('state')), dtype=T.float32)
            self.new_state_memory = T.tensor((d.item().get('nstate')), dtype=T.float32)
            self.action_memory = T.tensor((d.item().get('action')), dtype=T.int64)
            self.reward_memory = T.tensor((d.item().get('reward')), dtype=T.float32)
            self.terminal_memory = T.tensor((d.item().get('done')), dtype=T.bool)
            self.mem_cntr = self.mem_size
        else:
            # print("load offline data!!!!!")
            # d = np.load(self.dir)
            # # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])
            # self.state_memory = T.tensor((d.item().get('state')), dtype=T.float32)
            # self.new_state_memory = T.tensor((d.item().get('nstate')), dtype=T.float32)
            # self.action_memory = T.tensor((d.item().get('action')), dtype=T.int64)
            # self.new_action_memory = T.tensor((d.item().get('naction')), dtype=T.int64)
            # self.reward_memory = T.tensor((d.item().get('reward')), dtype=T.float32)
            # self.terminal_memory = T.tensor((d.item().get('done')), dtype=T.bool)
            # self.mem_cntr = self.mem_size
            #
            # self.state_memory = self.state_memory[:self.mem_size]
            # self.new_state_memory = self.new_state_memory[:self.mem_size]
            # self.action_memory = self.action_memory[:self.mem_size]
            # self.new_action_memory = self.new_action_memory[:self.mem_size]
            # self.reward_memory = self.reward_memory[:self.mem_size]
            # self.terminal_memory = self.terminal_memory[:self.mem_size]
            # self.mem_cntr = len(self.state_memory)-1

            d = np.load(self.dir)
            # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])
            self.state_memory = T.zeros((self.mem_size, self.in_channels, self.input_dims, self.input_dims), dtype=T.float32)
            self.new_state_memory = T.zeros((self.mem_size, self.in_channels, self.input_dims, self.input_dims), dtype=T.float32)
            self.action_memory = T.zeros(self.mem_size, dtype=T.int64)
            self.new_action_memory = T.zeros(self.mem_size, dtype=T.int64)
            self.reward_memory = T.zeros(self.mem_size, dtype=T.float32)
            self.terminal_memory = T.zeros(self.mem_size, dtype=T.bool)

            temp = T.tensor((d.item().get('state')), dtype=T.float32)

            self.state_memory[:len(temp), :] = T.tensor((d.item().get('state')), dtype=T.float32)
            self.new_state_memory[:len(temp), :] = T.tensor((d.item().get('nstate')), dtype=T.float32)
            self.action_memory[:len(temp)] = T.tensor((d.item().get('action')), dtype=T.int64)
            self.new_action_memory[:len(temp)] = T.tensor((d.item().get('naction')), dtype=T.int64)
            self.reward_memory[:len(temp)] = T.tensor((d.item().get('reward')), dtype=T.float32)
            self.terminal_memory[:len(temp)] = T.tensor((d.item().get('done')), dtype=T.bool)
            self.mem_cntr = len(temp)

