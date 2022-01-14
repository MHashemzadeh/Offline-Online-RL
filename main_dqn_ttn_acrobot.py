#import gym
import numpy as np
from dqn_agent import DQNAgent
from ttn_agent_online import TTNAgent_online
from mix_ttn_agent_online_offline import TTNAgent_online_offline_mix
# from utils import plot_learning_curve, make_env
import os
import sys
import random
import gym
import time
#import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
import copy
import torch as T
import argparse
# import datetime as date
from datetime import datetime
from replay_memory import ReplayBuffer
# from numba import jit
# import nvidia_smi



# @jit(target='cuda')
def main(alg_type, hyper_num, data_length_num, mem_size, num_rep, offline, fqi_rep_num, num_step_ratio_mem, en):

    data_lengths = [mem_size, 6000, 10000, 20000, 30000]
    data_length = data_lengths[data_length_num]

    fqi_reps = [1, 10, 50, 100, 300]
    fqi_rep = fqi_reps[fqi_rep_num]

    alg = alg_type

    gamma = 0.99

    num_steps = num_step_ratio_mem  # 200000


    ## select environment
    if en == "Mountaincar":
        env = gym.make('MountainCar-v0')
        input_dim = env.observation_space.shape[0]
        num_act = 3
    elif en == "Acrobot":
        env = gym.make('Acrobot-v1')
        input_dim = env.observation_space.shape[0]
        num_act = 3
    elif en == "LunarLander":
        env = gym.make('LunarLander-v2')
        input_dim = env.observation_space.shape[0]
        num_act = 4
    elif en == "cartpole":
        env = gym.make('CartPole-v0')
        input_dim = env.observation_space.shape[0]
        num_act = 2

    rand_seed = num_rep * 32  # 332
    env.seed(rand_seed)
    T.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    ## normolize states
    def process_state(state, normalize=True):
        # states = np.array([state['position'], state['vel']])

        if normalize:
            if en == "Acrobot":
                states = np.array([state[0], state[1], state[2], state[3], state[4], state[5]])
                states[0] = (states[0] + 1) / (2)
                states[1] = (states[1] + 1) / (2)
                states[2] = (states[2] + 1) / (2)
                states[3] = (states[3] + 1) / (2)
                states[4] = (states[4] + (4*np.pi)) / (2*4*np.pi)
                states[5] = (states[5] + (9*np.pi)) / (2*4*np.pi)
            elif en == "LunarLander":
                states = np.array([state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]])
                mean = [0, 0.9, 0, -0.6, 0, 0, 0, 0]
                deviation = [0.35, 0.6, 0.7, 0.6, 0.5, 0.5, 1.0, 1.0]
                states[0] = (states[0] - mean[0]) / (deviation[0])
                states[1] = (states[1] - mean[1]) / (deviation[1])
                states[2] = (states[2] - mean[2]) / (deviation[2])
                states[3] = (states[3] - mean[3]) / (deviation[3])
                states[4] = (states[4] - mean[4]) / (deviation[4])
                states[5] = (states[5] - mean[5]) / (deviation[5])

            elif en=="cartpole":
                states = np.array([state[0], state[1], state[2], state[3]])
                states[0] = states[0]
                states[1] = states[1]
                states[2] = states[2]
                states[3] = states[3]

            elif en=="Mountaincar":
                states = np.array([state[0], state[1]])
                states[0] = (states[0] + 1.2) / (0.6 + 1.2)
                states[1] = (states[1] + 0.07) / (0.07 + 0.07)

        return states




    #dqn:
    hyper_sets_DQN = OrderedDict([("nn_lr", np.power(10, [-3.25, -3.5, -3.75, -4.0, -4.25])),
                                      ("eps_decay_steps", [10000, 20000, 40000]),
                                     ])

    ## DQN
    q_nnet_params = {"update_fqi_steps": 50000,
                     "num_actions": num_act,
                     "eps_init": 1.0,
                     "eps_final": 0.01,
                     "batch_size": 32,  # TODO try big batches
                     "replay_memory_size": mem_size,
                     "replay_init_size": 5000,
                     "update_target_net_steps": 1000
                     }

    ## TTN
    nnet_params = {"loss_features": 'semi_MSTDE', #"next_reward_state", next_state, semi_MSTDE
                   "beta1": 0.0,
                   "beta2": 0.99,
                   "eps_init": 1.0,
                   "eps_final": 0.01,
                   "num_actions": num_act,
                   "replay_memory_size": mem_size,  # 200000 if using ER
                   "replay_init_size": 5000,
                   "batch_size": 32,
                   "fqi_reg_type": "prev",# "l2" or "prev"
                    }
    ## TTN
    hyper_sets_lstdq = OrderedDict([("nn_lr", np.power(10, [-4.0])),  #[-2.0, -2.5, -3.0, -3.5, -4.0]
                                    ("reg_A",  [10, 20, 30, 50, 70, 100, 2, 3, 5, 8, 1, 0, 0.01, 0.002, 0.0003, 0.001, 0.05]),  # [0.0, -1.0, -2.0, -3.0] can also do reg towards previous weights
                                    ("eps_decay_steps", [1]),
                                    ("update_freq", [1000]),
                                    ("data_length", [data_length]),
                                    ("fqi_rep", [fqi_rep]),
                                    ])





    if alg == 'fqi':
        log_file = "mem_size_control_num_step_{}_{}_lenght_{}_date_{}_hyper_{}".format(num_step_ratio_mem, alg, data_length, datetime.today().strftime("%d_%m_%Y"), hyper_num)
    if alg == 'dqn':
        log_file = "mem_size_control_{}_date_{}_hyper_{}".format(alg, datetime.today().strftime("%d_%m_%Y"), hyper_num)



    if alg in ("fqi"):
        hyper_sets = hyper_sets_lstdq
        TTN = True
    elif alg == "dqn":
        hyper_sets = hyper_sets_DQN
        TTN = False
    #
    hyperparams_all = list(itertools.product(*list(hyper_sets.values())))
    hyperparams = hyperparams_all

    with open(log_file + ".txt", 'w') as f:
        print("Start! Seed: {}".format(rand_seed), file=f)



    times = []


    start_time = time.perf_counter()
    index_run = 0

    # num_steps = 200000
    prev_action_flag = 1
    num_repeats = num_rep # 10
    # TTN = True

    # for hyper in hyperparams:
    # run the algorithm
    hyper = hyperparams[hyper_num]
    hyperparam_returns = []
    hyperparam_values = []
    hyperparam_episodes = []
    data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])


    if alg in ('lstdq', 'fqi'):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("reg_A", hyper[1]),
                              ("eps_decay_steps", hyper[2]),
                              ("update_freq", hyper[3]),
                              ("data_length", hyper[4]),
                              ("fqi_rep", hyper[5]),
                              ])

        print(hyper[0], hyper[1], hyper[2])
        with open(log_file + ".txt", 'a') as f:
            for par in params:
                if par in ("nn_lr", "reg_A", "eps_decay_steps", "update_freq", "data_length"):
                    print("hyper_parameter:" + par + ":{} ".format(params[par]), file=f)

    elif alg in ("dqn"):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("eps_decay_steps", hyper[1])])
        with open(log_file + ".txt", 'a') as f:
            for par in params:
                if par in ("nn_lr", "eps_decay_steps"):
                    print("hyper_parameter:" + par + ":{} ".format(params[par]), file=f)




    for rep in range(num_repeats):

        with open(log_file + ".txt", 'a') as f:
            print("new run for the current setting".format(rep), file=f)

        if TTN:
            nn = TTNAgent_online(gamma, nnet_params=nnet_params, other_params=params, input_dims=input_dim, num_units_rep=128)
        else:
            nn = DQNAgent(gamma, q_nnet_params, params, input_dims=input_dim)

        # saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

        # populate the replay buffer with some number of transitions
        if alg == 'dqn' or (TTN and nnet_params['replay_memory_size'] > 0):
            print("initialize buffer")
            frame_history = []
            prev_state = None
            prev_action = None
            done = 0
            state_unnormal = env.reset()
            state = process_state(state_unnormal)
            # populate the replay buffer
            while nn.memory.mem_cntr < 10000: #nnet_params["replay_init_size"]:
                discount = gamma

                if done:
                    # print("Game over", val, "\tStep", i)
                    # print('game')
                    val = 0.0
                    discount = 0.0
                    state_unnormal = env.reset()
                    state = process_state(state_unnormal)

                # get action
                action = np.random.randint(0, num_act)

                # add to buffer
                if prev_state is not None:
                    if prev_action_flag == 1:
                        nn.memory.store_transition_withnewaction(prev_state, np.squeeze(prev_action), reward, state,
                                                                 np.squeeze(action), int(done))
                    else:
                        nn.memory.store_transition(prev_state, np.squeeze(prev_action), reward, state, int(done))

                # do one step in the environment
                new_state_unnormal, reward, done, info = env.step(np.squeeze(action))  # action is a 1x1 matrix
                new_state = process_state(new_state_unnormal)
                # update saved states
                prev_state = state
                prev_action = action
                state = new_state

        data['state'] = nn.memory.state_memory
        data['action'] = nn.memory.action_memory
        data['reward'] = nn.memory.reward_memory
        data['nstate'] = nn.memory.new_state_memory
        data['naction'] = nn.memory.new_action_memory
        data['done'] = nn.memory.terminal_memory

        # np.save(
        #     "ndata_acrobot_rnd_1000_1_0_05_epsilon01_mem_size_{}_{}_{}_lenght_{}_date_{}_hyper_{}_run_{}".format(mem_size,
        #                                                                                                 log_image_str,
        #                                                                                                 alg,
        #                                                                                                 data_length,
        #                                                                                                 datetime.today().strftime(
        #                                                                                                     "%d_%m_%Y"),
        #                                                                                                 hyper_num,
        #                                                                                                 rep), data)

        prev_state = None
        prev_action = None
        run_returns = []
        run_episodes = []
        run_values = []
        run_losses = []
        done = 0
        episodes = 0
        val = 0.0
        state_unnormal = env.reset()
        state = process_state(state_unnormal)
        ctr = 0

        # frame_history = []
        start_run_time = time.perf_counter()
        index_run += 1
        episode_length = 0
        ch = 0
        # print("Run", index_run)
        for i in range(num_steps):

            discount = gamma
            episode_length += 1

            if done:

                with open(log_file + ".txt", 'a') as f:
                    print('time:', datetime.today().strftime("%H_%M_%S"),'return', '{}'.format(int(val)).ljust(4), i, len(run_returns),
                          "\t{:.2f}".format(np.mean(run_returns[-100:])), file=f)

                run_returns.append(val)
                run_episodes.append(i)
                if TTN:
                    print(episodes, i, round(val, 2), round(np.mean(run_returns[-100:]), 3), episode_length ) #nn.lin_values.data
                else:
                    print(episodes, i, round(val, 2), round(np.mean(run_returns[-100:]), 3), episode_length )
                # avg_returns.append(np.mean(all_returns[-100:]))
                val = 0.0
                discount = 0.0
                episodes += 1
                episode_length = 0
                state_unnormal = env.reset()
                state = process_state(state_unnormal)

            ## Get action
            if TTN:
                action = nn.choose_action(state)
                q_values = nn.lin_values

            else:  # DQN
                action, q_values = nn.choose_action(state)


            # run_values.append(T.mean(q_values.data, axis=1)[0].detach())

            # update parameters
            if prev_state is not None:
                # prev_state_normal = process_state(prev_state)
                # state_normal = process_state(state)
                if prev_action_flag == 1:
                    if reward > 0:
                        for i in range(1):
                            nn.memory.store_transition_withnewaction(prev_state, np.squeeze(prev_action), reward, state, np.squeeze(action), int(done))
                            ctr += 1
                    else:
                        nn.memory.store_transition_withnewaction(prev_state, np.squeeze(prev_action), reward, state,
                                                                 np.squeeze(action), int(done))
                        ctr += 1
                else:
                    if reward > 0:
                        for i in range(1):
                            nn.memory.store_transition(prev_state, np.squeeze(prev_action), reward, state, int(done))
                            ctr += 1
                    else:
                        nn.memory.store_transition(prev_state, np.squeeze(prev_action), reward, state, int(done))
                        ctr += 1

                if TTN:
                    if reward == 1000:
                        ch = 1
                    if ch == 0:
                        loss = nn.learn()
                        print("learning")

                else:  # DQN
                    loss = nn.learn()

                # run_losses.append(loss.detach())

            # do one step in the environment
            new_state_unnormal, reward, done, info = env.step(np.array(action))  # action is a 1x1 matrix
            new_state = process_state(new_state_unnormal)

            # update saved states
            prev_state = state
            prev_action = action
            state = new_state  # action is a 1x1 matrix
            state_unnormal = new_state_unnormal
            reward = reward
            # if reward> 0:
            #     print(reward)
            val += reward


            # np.save("Returns_mc_original_num_step_{}_mem_size_{}_{}_{}_lenght_{}_date_{}_hyper_{}".format(num_step_ratio_mem, mem_size, log_image_str, alg, data_length, datetime.today().strftime("%d_%m_%Y"), hyper_num),
            #         run_returns)

        hyperparam_returns.append(run_returns)
        hyperparam_episodes.append(run_episodes)
        hyperparam_values.append(run_values)

        # np.save("fqi_mc_hyperparam_returns"+str(data_length)+str(hyper_num)+str(mem_size), hyperparam_returns)
        # np.save("Returns_mc_num_step_{}_mem_size_{}_{}_{}_lenght_{}_date_{}_hyper_{}".format(num_step_ratio_mem, mem_size, log_image_str, alg, data_length,datetime.today().strftime("%d_%m_%Y"), hyper_num), hyperparam_returns)

        data['state'] = nn.memory.state_memory
        data['action'] = nn.memory.action_memory
        data['reward'] = nn.memory.reward_memory
        data['nstate'] = nn.memory.new_state_memory
        data['naction'] = nn.memory.new_action_memory
        data['done'] = nn.memory.terminal_memory
        # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])

        np.save("ndata_ll_1000_mem_size_{}_{}_lenght_{}_date_{}_hyper_{}_run_{}".format(mem_size, alg, data_length,datetime.today().strftime("%d_%m_%Y"), hyper_num, rep), data)

        print(episodes)
        print(data['state'].size(), mem_size)
        print(ctr)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, default='fqi')
    parser.add_argument('--hyper_num', type=int, default= 15 )
    parser.add_argument('--data_length_num', type=int, default=0)
    parser.add_argument('--mem_size', type=int, default=20000)
    parser.add_argument('--num_rep', type=int, default=1)
    parser.add_argument('--offline', type=bool, default=False)
    parser.add_argument('--fqi_rep_num', type=int, default=0)
    parser.add_argument('--num_step_ratio_mem', type=int, default=70000)
    parser.add_argument('--en', type=str, default='cartpole')


    args = parser.parse_args()

    main(args.algo, args.hyper_num, args.data_length_num, args.mem_size, args.num_rep, args.offline, args.fqi_rep_num, args.chunk_num, args.num_step_ratio_mem, args.en)




#15, -3 .... 12, -4,....  15, -4
