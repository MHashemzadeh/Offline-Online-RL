# import gym
import numpy as np
from dqn_agent import DQNAgent
# from ttn_agent_online import TTNAgent_online
#from ttn_agent_online_tc import TTNAgent_online_tc #TODO: You need to add this file to the git.
from mix_ttn_agent_online_offline import TTNAgent_online_offline_mix
# from utils import plot_learning_curve, make_env
import os
import sys
import random
import gym
import time
# import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
import copy
import torch as T
import argparse
# import datetime as date
from datetime import datetime
from replay_memory import ReplayBuffer
from utils.env_utils import process_state_constructor
from envs.env_constructor import get_env
from pathlib import Path
import json

from scipy import sparse

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# np_load_old = np.load

# modify the default parameters of np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# env_ = "catcher"
#
# if env_ == "catcher":
#     from ple.games.catcher import Catcher
# else:
#     import gym

######################################################
def train_online(data_dir, alg_type, hyper_num, data_length_num, mem_size, num_rep, offline, fqi_rep_num, num_step_ratio_mem, en,
          feature, method_sarsa,num_epi_per_itr,
                         fqi_reg_type, initial_batch, rnd, status):

    print("online", hyper_num, en, mem_size)

    data_lengths = [mem_size, 6000, 10000, 20000, 30000]
    data_length = data_lengths[data_length_num]

    fqi_reps = [1, 10, 50, 100, 300]
    fqi_rep = fqi_reps[fqi_rep_num]

    if feature == 'tc':
        tilecoding = 1
    else:
        tilecoding = 0

    alg = alg_type

    gamma = 0.99

    num_steps = num_step_ratio_mem  # 200000
    rand_seed = num_rep * 32

    ## select environment
    env, input_dim, num_act = get_env(en, seed=rand_seed)

    ### Normalize State
    process_state = process_state_constructor(en)

    # rand_seed = num_rep * 32  # 332 #TODO: T
    # env.seed(rand_seed) 
    # T.manual_seed(rand_seed)
    # np.random.seed(rand_seed)

    # dqn:
    hyper_sets_DQN = OrderedDict([("nn_lr", np.power(10, [-3.25, -3.5, -3.75, -4.0, -4.25])),
                                  ("eps_decay_steps", [1, 20000, 40000]),
                                  ])

    ## DQN
    q_nnet_params = {"update_fqi_steps": 50000,
                     "num_actions": num_act,
                     "eps_init": 1.0,
                     "eps_final": 0.01,
                     "batch_size": 32,  # TODO try big batches
                     "replay_memory_size": mem_size,
                     "replay_init_size": 5000,
                     "update_target_net_steps": 100
                     }

    ## TTN
    nnet_params = {"loss_features": 'semi_MSTDE',  # "next_reward_state", next_state, semi_MSTDE
                   "beta1": 0.0,
                   "beta2": 0.99,
                   "eps_init": 1.0,
                   "eps_final": 0.01,
                   "num_actions": num_act,
                   "replay_memory_size": mem_size,  # 200000 if using ER
                   "replay_init_size": 5000,
                   "batch_size": 32,
                   "fqi_reg_type": fqi_reg_type,  # "l2" or "prev"
                   }

    ## TTN
    hyper_sets_lstdq = OrderedDict([("nn_lr", np.power(10, [-3.0, -3.5, -4.0]).tolist()), 
                                    ("reg_A", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]),
                                    ("eps_decay_steps", [1]),
                                    ("update_freq", [1000]),  # default 1000
                                    ("data_length", [data_length]),
                                    ("fqi_rep", [fqi_rep]),
                                    # Data Augmentation Params
                                    ("data_aug_type", ['ras']),
                                    ("data_aug_prob", [0.0]),  #[0.0, 0.01, 0.05, 0.1]
                                    ("data_aug_loc", ['rep']), #['rep', 'fqi', 'both']
                                    ("random_shift_pad", [4]),
                                    ("ras_alpha", [0.9]), #0.6 , 0.8
                                    ("ras_beta", [1.1]),   #1.2 , 1.4
                                    # Input Transformations Params
                                    ("trans_type", ['none', 'sparse', 'sparse_random']),
                                    ("new_feat_dim", [32, 64, 128]),
                                    ("sparse_density", [0.1, 0.5])
                                    ])

    with open('Online//Online_Original_hyper_sets_lstdq.json', 'w') as fp:
        fp.write(json.dumps(hyper_sets_lstdq, indent=4))
    fp.close()
    
    if alg in ("fqi"):
        hyper_sets = hyper_sets_lstdq
        TTN = True
    elif alg == "dqn":
        hyper_sets = hyper_sets_DQN
        TTN = False
    #
    hyperparams_all = list(itertools.product(*list(hyper_sets.values())))
    hyperparams = hyperparams_all

    # Removing redundant experiments for data_aug_prob = 0
    hyperparams_filtered = []
    count = 0
    for i in range(len(hyperparams)):
        if(i%18 not in [1,2,3,4,5]):
            hyperparams_filtered.append(hyperparams[i])
            count += 1

    print(f"Number of Hyperparams: {count}")
    hyperparams = hyperparams_filtered

    # write to a file to maintain the indexes
    with open("Online//Online-sweep-parameters.txt", "w") as fp:
        for ix in range(len(hyperparams)):
            to_write = str(ix) + " " + '-'.join([str(x) for x in hyperparams[ix]])
            fp.write(to_write + "\n")
    fp.close()

    times = []
    start_time = time.perf_counter()

    prev_action_flag = 1
    num_repeats = num_rep  # 10

    hyper = hyperparams[hyper_num]

    #TODO: Add the following params too for sweep
    #TODO: num_updates_pre_train 
    #TODO: mem_size
    #TODO: type_of_augmentation (rep, fqi, both)
    #TODO: use traget_network 

    output_dir_path = "Online//Training_{}_online_env_{}_mem_size_{}_hyper_num_{}_date_{}".format(
                                                                                                 alg_type,
                                                                                                 en,
                                                                                                 mem_size,
                                                                                                 hyper_num,
                                                                                                 datetime.today().strftime("%d_%m_%Y")
                                                                                                )

    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    files_name = "{}//".format(output_dir_path)


    if rnd:
        files_name = files_name+'_rnd'
    if initial_batch:
        files_name = files_name + '_initialbatch_'

    if alg == 'fqi':
        log_file = files_name+str(alg)
    if alg == 'dqn':
        log_file = files_name+str(alg)


    data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])

    if alg in ('lstdq', 'fqi'):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("reg_A", hyper[1]),
                              ("eps_decay_steps", hyper[2]),
                              ("update_freq", hyper[3]),
                              ("data_length", hyper[4]),
                              ("fqi_rep", hyper[5]),
                              # Data Augmentation Params
                              ("data_aug_type", hyper[6]),
                              ("data_aug_prob", hyper[7]),
                              ("data_aug_loc", hyper[8]),
                              ("random_shift_pad", hyper[9]),
                              ("ras_alpha", hyper[10]), #0.6 , 0.8
                              ("ras_beta", hyper[11]),   #1.2 , 1.4
                              # Input Transformation Params
                              ("trans_type", hyper[12]),
                              ("new_feat_dim", hyper[13]),
                              ("sparse_density", hyper[14])
                              ])
        print(f"Params: {params}")
        # print(f"Nnet params: {nnet_params}")

    elif alg in ("dqn"):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("eps_decay_steps", hyper[1])])



    # saved_state_list_all = np.load(starting_state_path)  # np.load starting states

    # #######                        Creating Identity matrix                      #######
    if(params["trans_type"] == "none"):
        sparse_matrix = np.identity(input_dim)

    # #######                        Creating Sparse matrix                      #######
    elif(params["trans_type"] == "sparse"):
        sparse_matrix = np.random.choice(2, size=(input_dim, params["new_feat_dim"]), p=[1 - params["sparse_density"], params["sparse_density"]])
        #making sure that original features occur atleast once in the new features
        for row in range(sparse_matrix.shape[0]):
            sparse_matrix[row][row] = 1.0

    #######                        Creating Sparse random matrix                      #######
    elif(params["trans_type"] == "sparse_random"):
        
        sparse_matrix = np.array(sparse.random(input_dim, params["new_feat_dim"], density=params["sparse_density"], data_rvs=np.random.randn, dtype=np.float32).todense())
        sparse_matrix = (0.35 * sparse_matrix) / (np.sqrt(input_dim))
        #making sure that original features occur atleast once in the new features
        for row in range(sparse_matrix.shape[0]):
            sparse_matrix[row][row] = 1.0

    print(f"sparse_matrix: {sparse_matrix} , sparse_matrix.shape: {sparse_matrix.shape}")

    hyperparam_returns = []
    hyperparam_values = []
    hyperparam_episodes = []
    hyperparam_losses = []

    hyperparam_avgreturns = []
    hyperparam_avgvalues = []
    hyperparam_avgepisodes = []
    hyperparam_avgepisodevalues = []

    hyperparam_stdreturns= []
    hyperparam_stdvalues = []
    hyperparam_stdepisodes = []
    hyperparam_stdepisodevalues = []

    hyperparam_final_avgreturns = []
    hyperparam_final_avgvalues = []
    hyperparam_final_avgepisodes = []
    hyperparam_final_avgepisodevalues = []

    hyperparam_final_stdreturns = []
    hyperparam_final_stdvalues = []
    hyperparam_final_stdepisodes = []
    hyperparam_final_stdepisodevalues = []

    for rep in range(num_repeats):


        rand_seed = rep * 32  # 332
        
        if en != "catcher":
            env.seed(rand_seed)

        T.manual_seed(rand_seed)
        np.random.seed(rand_seed)

        with open(log_file + ".txt", 'w') as f:
            print("Start! Seed: {}".format(rand_seed), file=f)

        # saved_state_list = saved_state_list_all[rep * num_epi_per_itr:rep * num_epi_per_itr + num_epi_per_itr]

        #############################################
        start_run_time = time.perf_counter()

        if TTN:

            nn = TTNAgent_online_offline_mix(gamma, nnet_params=nnet_params, other_params=params, sparse_matrix=sparse_matrix,
                                             input_dims=input_dim, num_units_rep=128, dir=data_dir, offline=offline,
                                             num_tiling=16, num_tile=4, method_sarsa=method_sarsa,
                                             tilecoding=tilecoding, status=status)

        else:
            nn = DQNAgent(gamma, q_nnet_params, params, input_dims=input_dim, dir=data_dir, status = status)

        if initial_batch == False:

            if alg == 'dqn' or (TTN and nnet_params['replay_memory_size'] > 0):
                print("initialize buffer")
                frame_history = []
                prev_state = None
                prev_action = None
                done = 0

                if en != "catcher":
                    state_unnormal = env.reset()
                else:
                    state_unnormal = env.reset_game()

                state = process_state(state_unnormal)
                # populate the replay buffer
                while nn.memory.mem_cntr < nnet_params["replay_init_size"]:

                    if done:
                        if en != "catcher":
                            state_unnormal = env.reset()
                        else:
                            state_unnormal = env.reset_game()
                            
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
                    prev_state = state
                    prev_action = action
                    state = new_state

        else:
            nn.memory.offline = True
            nn.memory.load_mem()


        prev_state = None
        prev_action = None

        run_returns = []
        run_vals = []
        run_episode_length = []
        run_avg_episode_values = []

        run_losses = []



        for itr in range(1):

            # # do update before running the agent
            # for j in range(num_updates_pretrain):
            #     if TTN:
            #         nn.learn_pretrain()
            #
            #     else:  # DQN
            #         loss = nn.learn()

            run_avgreturns = []
            run_avgvals = []
            run_avgepisode_length = []

            done = 0
            episodes = 0
            val = 0.0
            q_values_episode = 0
            ret = 0
            episode_length = 0
            count_10epi = 0

            if en != "catcher":
                env.reset()
            else:
                env.reset_game()

            # game = env.unwrapped
            # env.state = saved_state_list[count_10epi]
            # state_unnormal = env.state
            # state = process_state(state_unnormal)

            # state_unnormal = env.unwrapped.state
            if en != "catcher":
                state_unnormal = env.reset()
            else:
                state_unnormal = env.reset_game()

            state = process_state(state_unnormal)

            start_run_time = time.perf_counter()

            i = 0

            while episodes < num_epi_per_itr + 1:

                i += 1

                if done:

                    run_returns.append(ret)
                    run_vals.append(val)
                    run_episode_length.append(episode_length)
                    run_avg_episode_values.append((q_values_episode / episode_length))

                    run_avgreturns.append(ret)
                    run_avgvals.append(val)
                    run_avgepisode_length.append(episode_length)

                    print(episodes, i, round(val, 2), ret, "number episode from 10:", count_10epi,
                          "avegar over 10 episode:", round(np.mean(run_avgreturns), 3),
                          "avegare return across last 100 episodes:", round(np.mean(run_returns[-100:]), 3),
                          "state values:", (q_values_episode / episode_length), episode_length)

                    count_10epi += 1
                    episode_length = 0
                    q_values_episode = 0
                    val = 0.0
                    ret = 0.0
                    i = 0
                    episodes += 1

                    if en != "catcher":
                        env.reset()
                    else:
                        env.reset_game()
                    # env.state = saved_state_list[count_10epi]
                    # state_unnormal = env.state
                    # state = process_state(state_unnormal)

                    # state_unnormal = env.unwrapped.state
                    if en != "catcher":
                        state_unnormal = env.reset()
                    else:
                        state_unnormal = env.reset_game()

                    state = process_state(state_unnormal)

                    if TTN:
                        nn.load_data()
                        nn.learn_pretrain()


                # Get action
                if TTN:
                    action = nn.choose_action(state)
                    q_values = nn.lin_values

                else:  # DQN
                    action, q_values = nn.choose_action(state)

                # run_values.append(T.mean(q_values.data, axis=1)[0].detach())
                q_values_episode += T.mean(q_values.data)

                # update parameters
                if prev_state is not None:
                    # prev_state_normal = process_state(prev_state)
                    # state_normal = process_state(state)
                    if prev_action_flag == 1:
                        nn.memory.store_transition_withnewaction(prev_state, np.squeeze(prev_action), reward, state,
                                                                 np.squeeze(action), int(done))
                    else:
                        nn.memory.store_transition(prev_state, np.squeeze(prev_action), reward, state, int(done))

                    if TTN:
                        loss = nn.learn()

                    else:  # DQN
                        loss = nn.learn()

                    # run_losses.append(loss.detach())

                # do one step in the environment
                episode_length += 1
                new_state_unnormal, reward, done, info = env.step(np.array(action))  # action is a 1x1 matrix
                new_state = process_state(new_state_unnormal)

                # update saved states
                prev_state = state
                prev_action = action
                state = new_state  # action is a 1x1 matrix
                val += reward

                if reward > 100 or reward== 100:
                    print(reward)

                ret += (gamma**i) * reward

            ## store average values at each iteration
            hyperparam_avgreturns.append(np.mean(run_avgreturns))
            hyperparam_avgvalues.append(np.mean(run_avgvals))
            hyperparam_avgepisodes.append(np.mean(run_avgepisode_length))
            hyperparam_avgepisodevalues.append(np.mean(run_avg_episode_values))

            hyperparam_stdreturns.append(np.std(run_avgreturns))
            hyperparam_stdvalues.append(np.std(run_avgvals))
            hyperparam_stdepisodes.append(np.std(run_avgepisode_length))
            hyperparam_stdepisodevalues.append(np.std(run_avg_episode_values))

        hyperparam_returns.append(run_returns)
        hyperparam_values.append(run_avg_episode_values)
        hyperparam_episodes.append(run_episode_length)

        np.save(files_name + 'hyperparam_avgreturns', hyperparam_avgreturns)
        np.save(files_name + 'hyperparam_avgepisodes', hyperparam_avgepisodes)
        np.save(files_name + 'hyperparam_avgepisodevalues', hyperparam_avgepisodevalues)

        np.save(files_name + 'hyperparam_stdreturns', hyperparam_stdreturns)
        np.save(files_name + 'hyperparam_stdepisodes', hyperparam_stdepisodes)
        np.save(files_name + 'hyperparam_stdepisodevalues', hyperparam_stdepisodevalues)

        np.save( files_name + 'hyperparam_returns', hyperparam_returns)
        np.save( files_name + 'hyperparam_values', hyperparam_values)
        np.save( files_name + 'hyperparam_episodes', hyperparam_episodes)

        hyperparam_final_avgreturns.append(hyperparam_returns)
        hyperparam_final_avgvalues.append(hyperparam_values)
        hyperparam_final_avgepisodes.append(hyperparam_episodes)

    # np.save(files_name + 'hyperparam_final_avgreturns', np.mean(hyperparam_final_avgreturns, axis=0))
    # np.save(files_name + 'hyperparam_final_avgvalues', np.mean(hyperparam_final_avgvalues, axis=0))
    # np.save(files_name + 'hyperparam_final_avgepisodes', np.mean(hyperparam_final_avgepisodes, axis=0))
    #
    # np.save(files_name + 'hyperparam_final_stdreturns', np.std(hyperparam_final_avgreturns, axis=0))
    # np.save(files_name + 'hyperparam_final_stdvalues', np.std(hyperparam_final_avgvalues, axis=0))
    # np.save(files_name + 'hyperparam_final_stdepisodes', np.std(hyperparam_final_avgepisodes, axis=0))

    np.save(files_name + 'hyperparam_final_avgreturns', np.mean(hyperparam_returns, axis=0))
    np.save(files_name + 'hyperparam_final_avgvalues', np.mean(hyperparam_values, axis=0))
    np.save(files_name + 'hyperparam_final_avgepisodes', np.mean(hyperparam_episodes, axis=0))

    np.save(files_name + 'hyperparam_final_stdreturns', np.std(hyperparam_returns, axis=0))
    np.save(files_name + 'hyperparam_final_stdvalues', np.std(hyperparam_values, axis=0))
    np.save(files_name + 'hyperparam_final_stdepisodes', np.std(hyperparam_episodes, axis=0))



######################################################
def train_offline_online(data_dir, alg_type, hyper_num, data_length_num, mem_size, num_rep, offline,
                 fqi_rep_num, num_step_ratio_mem, en,
                 feature, method_sarsa, num_epi_per_itr,
                 fqi_reg_type, initial_batch, rnd, num_updates_pretrain, epsilon_stop_training, status):

    print("offline_online", hyper_num, en,mem_size)

    data_lengths = [mem_size, 6000, 10000, 20000, 30000]
    data_length = data_lengths[data_length_num]

    fqi_reps = [1, 10, 50, 100, 300]
    fqi_rep = fqi_reps[fqi_rep_num]

    if feature == 'tc':
        tilecoding = 1
    else:
        tilecoding = 0

    alg = alg_type

    gamma = 0.99

    num_steps = num_step_ratio_mem  # 200000

    rand_seed = num_rep * 32

    ## select environment
    env, input_dim, num_act = get_env(en, seed=rand_seed)


    ### Normalize State
    process_state = process_state_constructor(en)

    # rand_seed = num_rep * 32  # 332
    # env.seed(rand_seed)
    # T.manual_seed(rand_seed)
    # np.random.seed(rand_seed)

    # dqn:
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
    nnet_params = {"loss_features": 'semi_MSTDE',  # "next_reward_state", next_state, semi_MSTDE
                   "beta1": 0.0,
                   "beta2": 0.99,
                   "eps_init": 1.0,
                   "eps_final": 0.01,
                   "num_actions": num_act,
                   "replay_memory_size": mem_size,  # 200000 if using ER
                   "replay_init_size": 5000,
                   "batch_size": 32,
                   "fqi_reg_type": fqi_reg_type,  # "l2" or "prev"
                   }

    ## TTN
    hyper_sets_lstdq = OrderedDict([("nn_lr", np.power(10, [-3.0, -3.5, -4.0]).tolist()), 
                                    ("reg_A", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]),
                                    ("eps_decay_steps", [1]),
                                    ("update_freq", [1000]),  # default 1000
                                    ("data_length", [data_length]),
                                    ("fqi_rep", [fqi_rep]),
                                    # Data Augmentation Params
                                    ("data_aug_type", ['ras']),
                                    ("data_aug_prob", [0.0]),  #[0.0, 0.01, 0.05, 0.1]
                                    ("data_aug_loc", ['rep']), #['rep', 'fqi', 'both']
                                    ("random_shift_pad", [4]),
                                    ("ras_alpha", [0.9]), #0.6 , 0.8
                                    ("ras_beta", [1.1]),   #1.2 , 1.4
                                    # Input Transformations Params
                                    ("trans_type", ['none', 'sparse', 'sparse_random']),
                                    ("new_feat_dim", [32, 64, 128]),
                                    ("sparse_density", [0.1, 0.5])
                                    ])

    with open('Offline-online//Offline_online_Original_hyper_sets_lstdq.json', 'w') as fp:
        fp.write(json.dumps(hyper_sets_lstdq, indent=4))

    if alg in ("fqi"):
        hyper_sets = hyper_sets_lstdq
        TTN = True
    elif alg == "dqn":
        hyper_sets = hyper_sets_DQN
        TTN = False
    #
    hyperparams_all = list(itertools.product(*list(hyper_sets.values())))
    hyperparams = hyperparams_all

    # # Removing redundant experiments for data_aug_prob = 0
    hyperparams_filtered = []
    count = 0
    for i in range(len(hyperparams)):
        if(i%18 not in [1,2,3,4,5]):
            hyperparams_filtered.append(hyperparams[i])
            count += 1

    print(f"Number of Hyperparams: {count}")
    hyperparams = hyperparams_filtered

    # write to a file to maintain the indexes
    with open("Offline-online//Offline-online-sweep-parameters.txt", "w") as fp:
        for ix in range(len(hyperparams)):
            to_write = str(ix) + " " + '-'.join([str(x) for x in hyperparams[ix]])
            fp.write(to_write + "\n")
    fp.close()
    
    
    times = []
    start_time = time.perf_counter()

    prev_action_flag = 1
    num_repeats = num_rep  # 10
    hyper = hyperparams[hyper_num]


    #TODO: Add the following params too for sweep
    #TODO: num_updates_pre_train 
    #TODO: mem_size
    #TODO: type_of_augmentation (rep, fqi, both)
    #TODO: use traget_network 


    output_dir_path = "Offline-online//Training_{}_offline_online_env_{}_mem_size_{}_hyper_num_{}_date_{}".format(
                                                                                                 alg_type,
                                                                                                 en,
                                                                                                 mem_size,
                                                                                                 hyper_num,
                                                                                                 datetime.today().strftime("%d_%m_%Y")
                                                                                                )

    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    files_name = "{}//".format(output_dir_path)


    if rnd:
        files_name = files_name + '_rnd'
    if initial_batch:
        files_name = files_name + '_initialbatch_'

    if alg == 'fqi':
        log_file = files_name + str(alg)
    if alg == 'dqn':
        log_file = files_name + str(alg)


    data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])

    if alg in ('lstdq', 'fqi'):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("reg_A", hyper[1]),
                              ("eps_decay_steps", hyper[2]),
                              ("update_freq", hyper[3]),
                              ("data_length", hyper[4]),
                              ("fqi_rep", hyper[5]),
                              # Data Augmentation Params
                              ("data_aug_type", hyper[6]),
                              ("data_aug_prob", hyper[7]),
                              ("data_aug_loc", hyper[8]),
                              ("random_shift_pad", hyper[9]),
                              ("ras_alpha", hyper[10]), #0.6 , 0.8
                              ("ras_beta", hyper[11]),   #1.2 , 1.4
                              # Input Transformation Params
                              ("trans_type", hyper[12]),
                              ("new_feat_dim", hyper[13]),
                              ("sparse_density", hyper[14])
                              ])
        print(f"Params: {params}")
        # print(f"Nnet params: {nnet_params}")

    elif alg in ("dqn"):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("eps_decay_steps", hyper[1])])

    # saved_state_list_all = np.load(starting_state_path)  # np.load starting states

    # #######                        Creating Identity matrix                      #######
    if(params["trans_type"] == "none"):
        sparse_matrix = np.identity(input_dim)

     # #######                        Creating Sparse matrix                      #######
    elif(params["trans_type"] == "sparse"):
        sparse_matrix = np.random.choice(2, size=(input_dim, params["new_feat_dim"]), p=[1 - params["sparse_density"], params["sparse_density"]])
        #making sure that original features occur atleast once in the new features
        for row in range(sparse_matrix.shape[0]):
            sparse_matrix[row][row] = 1.0

    elif(params["trans_type"] == "sparse_random"):
        #######                        Creating Sparse random matrix                      #######
        sparse_matrix = np.array(sparse.random(input_dim, params["new_feat_dim"], density=params["sparse_density"], data_rvs=np.random.randn, dtype=np.float32).todense())
        sparse_matrix = (0.35 * sparse_matrix) / (np.sqrt(input_dim))
        #making sure that original features occur atleast once in the new features
        for row in range(sparse_matrix.shape[0]):
            sparse_matrix[row][row] = 1.0

    print(f"sparse_matrix: {sparse_matrix} , sparse_matrix.shape: {sparse_matrix.shape}")


    hyperparam_returns = []
    hyperparam_values = []
    hyperparam_episodes = []
    hyperparam_losses = []

    hyperparam_avgreturns = []
    hyperparam_avgvalues = []
    hyperparam_avgepisodes = []
    hyperparam_avgepisodevalues = []

    hyperparam_stdreturns = []
    hyperparam_stdvalues = []
    hyperparam_stdepisodes = []
    hyperparam_stdepisodevalues = []

    hyperparam_final_avgreturns = []
    hyperparam_final_avgvalues = []
    hyperparam_final_avgepisodes = []
    hyperparam_final_avgepisodevalues = []

    hyperparam_final_stdreturns = []
    hyperparam_final_stdvalues = []
    hyperparam_final_stdepisodes = []
    hyperparam_final_stdepisodevalues = []

    for rep in range(num_repeats):

        rand_seed = rep * 332 #32  # 332

        if en != "catcher":
            env.seed(rand_seed)

        # env.seed(rand_seed)
        T.manual_seed(rand_seed)
        np.random.seed(rand_seed)

        with open(log_file + ".txt", 'w') as f:
            print("Start! Seed: {}".format(rand_seed), file=f)

        # saved_state_list = saved_state_list_all[rep * num_epi_per_itr:rep * num_epi_per_itr + num_epi_per_itr]

        #############################################
        start_run_time = time.perf_counter()

        if TTN:

            nn = TTNAgent_online_offline_mix(gamma, nnet_params=nnet_params, other_params=params, sparse_matrix=sparse_matrix,
                                             input_dims=input_dim, num_units_rep=128, dir=data_dir, offline=offline,
                                             num_tiling=16, num_tile=4, method_sarsa=method_sarsa,
                                             tilecoding=tilecoding, status=status)

        else:
            nn = DQNAgent(gamma, q_nnet_params, params, input_dims=input_dim, dir=data_dir, status=status)

        if initial_batch == False:

            if alg == 'dqn' or (TTN and nnet_params['replay_memory_size'] > 0):
                print("initialize buffer")
                frame_history = []
                prev_state = None
                prev_action = None
                done = 0
                if en != "catcher":
                    state_unnormal = env.reset()
                else:
                    state_unnormal = env.reset_game()

                state = process_state(state_unnormal)
                # populate the replay buffer
                while nn.memory.mem_cntr < nnet_params["replay_init_size"]:

                    if done:
                        if en != "catcher":
                            state_unnormal = env.reset()
                        else:
                            state_unnormal = env.reset_game()

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
                    prev_state = state
                    prev_action = action
                    state = new_state

        else:
            nn.memory.offline = True
            nn.memory.load_mem()

        prev_state = None
        prev_action = None

        run_returns = []
        run_vals = []
        run_episode_length = []
        run_avg_episode_values = []

        run_losses = []

        for itr in range(1):

            ## do update on step offline before running the agent
            # Here pretraining step would be skipped if files already exist 
            # if not os.path.isfile("feature_{}_{}_{}_{}".format(alg, en, mem_size, num_updates_pretrain) + ".pt"):

            #Always pre-train in offline 
            if True or (not os.path.isfile("{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")):
                print("Pre-training")
                if TTN:
                    loss = nn.learn()

                else:  # DQN
                    loss = nn.learn()

                loss1 = loss
                diff_loss = 1
                t = 0

                # while t < num_updates_pretrain: # and diff_loss> epsilon_stop_training:
                #     for j in range(params["update_freq"] + 5):
                #         if TTN:
                #             loss = nn.learn()
                #
                #         else:  # DQN
                #             loss = nn.learn()
                #
                #     diff_loss = abs(loss-loss1)
                #     print(loss)
                #     print(diff_loss)
                #     loss1= loss
                #     t += 1

                # batch_size = 64
                for j in range(num_updates_pretrain): #num_updates_pretrain = num_epoch = 100
                    num_iteration_feature = int(mem_size / nnet_params['batch_size'])
                    shuffle_index = np.arange(nnet_params['replay_memory_size'])
                    np.random.shuffle(shuffle_index)

                    for itr in range(num_iteration_feature):

                        if TTN:
                            loss = nn.learn_nn_feature_fqi(itr, shuffle_index)
                            # print(loss)

                        else:
                            loss = nn.learn_nn_feature(itr, shuffle_index)
                            # print(loss)

                if TTN:
                    T.save(nn.q_eval.state_dict(), "{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")

                    T.save(nn.lin_weights, "{}//lin_weights_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")
                else:
                    T.save(nn.q_eval.state_dict(), "{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")

                    T.save(nn.q_next.state_dict(), "{}//feature_next_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")

            else:

                print("features and lin_weights already exists, skipping pretraining!")
                if TTN:
                    nn.q_eval.load_state_dict(T.load("{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt"))
                    nn.lin_weights = T.load("{}//lin_weights_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")
                else:
                    nn.q_eval.load_state_dict(T.load("{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt"))
                    nn.q_next.load_state_dict(
                        T.load("{}//feature_next_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt"))

            ##
            ## end of offline step

            run_avgreturns = []
            run_avgvals = []
            run_avgepisode_length = []

            done = 0
            episodes = 0
            val = 0.0
            q_values_episode = 0
            ret = 0
            episode_length = 0
            count_10epi = 0

            if en != "catcher":
                env.reset()
            else:
                env.reset_game()
            # game = env.unwrapped
            # env.state = saved_state_list[count_10epi]
            # state_unnormal = env.state
            # state = process_state(state_unnormal)

            # state_unnormal = env.unwrapped.state
            if en != "catcher":
                state_unnormal = env.reset()
            else:
                state_unnormal = env.reset_game()

            state = process_state(state_unnormal)

            start_run_time = time.perf_counter()

            i = 0

            while episodes < num_epi_per_itr + 1:

                i += 1

                if done:
                    run_returns.append(ret)
                    run_vals.append(val)
                    run_episode_length.append(episode_length)
                    run_avg_episode_values.append((q_values_episode / episode_length))

                    run_avgreturns.append(ret)
                    run_avgvals.append(val)
                    run_avgepisode_length.append(episode_length)

                    print(episodes, i, round(val, 2), ret, "number episode from 10:", count_10epi,
                          "avegar over 10 episode:", round(np.mean(run_avgreturns), 3),
                          "avegare return across last 100 episodes:", round(np.mean(run_returns[-100:]), 3),
                          "state values:", (q_values_episode / episode_length), episode_length)

                    count_10epi += 1
                    episode_length = 0
                    q_values_episode = 0
                    val = 0.0
                    ret = 0.0
                    i = 0
                    episodes += 1

                    if en != "catcher":
                        env.reset()
                    else:
                        env.reset_game()
                    # env.state = saved_state_list[count_10epi]
                    # state_unnormal = env.state
                    # state = process_state(state_unnormal)

                    # state_unnormal = env.unwrapped.state
                    if en != "catcher":
                        state_unnormal = env.reset()
                    else:
                        state_unnormal = env.reset_game()
                    state = process_state(state_unnormal)
                    if TTN:
                        nn.load_data()
                        nn.learn_pretrain()


                # Get action
                if TTN:
                    action = nn.choose_action(state)
                    q_values = nn.lin_values

                else:  # DQN
                    action, q_values = nn.choose_action(state)

                # run_values.append(T.mean(q_values.data, axis=1)[0].detach())
                q_values_episode += T.mean(q_values.data)

                # update parameters
                if prev_state is not None:
                    # prev_state_normal = process_state(prev_state)
                    # state_normal = process_state(state)
                    if prev_action_flag == 1:
                        nn.memory.store_transition_withnewaction(prev_state, np.squeeze(prev_action), reward, state,
                                                                 np.squeeze(action), int(done))
                    else:
                        nn.memory.store_transition(prev_state, np.squeeze(prev_action), reward, state, int(done))

                    if TTN:
                        loss = nn.learn()

                    else:  # DQN
                        loss = nn.learn()

                    # run_losses.append(loss.detach())

                # do one step in the environment
                episode_length += 1
                new_state_unnormal, reward, done, info = env.step(np.array(action))  # action is a 1x1 matrix
                new_state = process_state(new_state_unnormal)

                # update saved states
                prev_state = state
                prev_action = action
                state = new_state  # action is a 1x1 matrix
                val += reward
                ret += (gamma ** i) * reward

            ## store average values at each iteration
            hyperparam_avgreturns.append(np.mean(run_avgreturns))
            hyperparam_avgvalues.append(np.mean(run_avgvals))
            hyperparam_avgepisodes.append(np.mean(run_avgepisode_length))
            hyperparam_avgepisodevalues.append(np.mean(run_avg_episode_values))

            hyperparam_stdreturns.append(np.std(run_avgreturns))
            hyperparam_stdvalues.append(np.std(run_avgvals))
            hyperparam_stdepisodes.append(np.std(run_avgepisode_length))
            hyperparam_stdepisodevalues.append(np.std(run_avg_episode_values))

        hyperparam_returns.append(run_returns)
        hyperparam_values.append(run_avg_episode_values)
        hyperparam_episodes.append(run_episode_length)

        np.save(files_name + 'hyperparam_avgreturns', hyperparam_avgreturns)
        np.save(files_name + 'hyperparam_avgepisodes', hyperparam_avgepisodes)
        np.save(files_name + 'hyperparam_avgepisodevalues', hyperparam_avgepisodevalues)

        np.save(files_name + 'hyperparam_stdreturns', hyperparam_stdreturns)
        np.save(files_name + 'hyperparam_stdepisodes', hyperparam_stdepisodes)
        np.save(files_name + 'hyperparam_stdepisodevalues', hyperparam_stdepisodevalues)

        np.save(files_name + 'hyperparam_returns', hyperparam_returns)
        np.save(files_name + 'hyperparam_values', hyperparam_values)
        np.save(files_name + 'hyperparam_episodes', hyperparam_episodes)

        hyperparam_final_avgreturns.append(hyperparam_returns)
        hyperparam_final_avgvalues.append(hyperparam_values)
        hyperparam_final_avgepisodes.append(hyperparam_episodes)

    # np.save(files_name + 'hyperparam_final_avgreturns', np.mean(hyperparam_final_avgreturns, axis=0))
    # np.save(files_name + 'hyperparam_final_avgvalues', np.mean(hyperparam_final_avgvalues, axis=0))
    # np.save(files_name + 'hyperparam_final_avgepisodes', np.mean(hyperparam_final_avgepisodes, axis=0))
    #
    # np.save(files_name + 'hyperparam_final_stdreturns', np.std(hyperparam_final_avgreturns, axis=0))
    # np.save(files_name + 'hyperparam_final_stdvalues', np.std(hyperparam_final_avgvalues, axis=0))
    # np.save(files_name + 'hyperparam_final_stdepisodes', np.std(hyperparam_final_avgepisodes, axis=0))

    np.save(files_name + 'hyperparam_final_avgreturns', np.mean(hyperparam_returns, axis=0))
    np.save(files_name + 'hyperparam_final_avgvalues', np.mean(hyperparam_values, axis=0))
    np.save(files_name + 'hyperparam_final_avgepisodes', np.mean(hyperparam_episodes, axis=0))

    np.save(files_name + 'hyperparam_final_stdreturns', np.std(hyperparam_returns, axis=0))
    np.save(files_name + 'hyperparam_final_stdvalues', np.std(hyperparam_values, axis=0))
    np.save(files_name + 'hyperparam_final_stdepisodes', np.std(hyperparam_episodes, axis=0))


def train_offline(data_dir, alg_type, hyper_num, data_length_num, mem_size, num_rep, offline,
                 fqi_rep_num, num_step_ratio_mem, en,
                 feature, method_sarsa, num_epi_per_itr,
                 fqi_reg_type, initial_batch, rnd, num_updates_pretrain, epsilon_stop_training, status):


    print("offline", hyper_num, en, mem_size) #mem)_size default in offline experiment is 10K 

    data_lengths = [mem_size, 6000, 10000, 20000, 30000]
    data_length = data_lengths[data_length_num] #data_length_num default = 0 ; data_length = mem_size

    fqi_reps = [1, 10, 50, 100, 300]
    fqi_rep = fqi_reps[fqi_rep_num] #default fqi_rep_num = 0 ; fqi_reps = 1 ; number of fqi repeats

    if feature == 'tc':
        tilecoding = 1
    else:
        tilecoding = 0

    alg = alg_type #fqi

    gamma = 0.99

    num_steps = num_step_ratio_mem  # 70K for offline default

    rand_seed = num_rep * 32

    ## select environment
    env, input_dim, num_act = get_env(en, seed=rand_seed)

    ### Normalize State
    process_state = process_state_constructor(en)
    
    # rand_seed = num_rep * 32  # 332
    # env.seed(rand_seed)
    # T.manual_seed(rand_seed)
    # np.random.seed(rand_seed)

    # dqn:
    hyper_sets_DQN = OrderedDict([("nn_lr", np.power(10, [-3.25, -3.5, -3.75, -4.0, -4.25])),
                                  ("eps_decay_steps", [1, 20000, 40000]),
                                  ])

    ## DQN
    q_nnet_params = {"update_fqi_steps": 50000,
                     "num_actions": num_act,
                     "eps_init": 1.0,
                     "eps_final": 0.01,
                     "batch_size": 32,  # TODO try big batches
                     "replay_memory_size": mem_size,
                     "replay_init_size": 5000,
                     "update_target_net_steps": 1000,
                     }

    ## TTN
    nnet_params = {"loss_features": 'semi_MSTDE',  # "next_reward_state", next_state, semi_MSTDE
                   "beta1": 0.0,
                   "beta2": 0.99,
                   "eps_init": 1.0,
                   "eps_final": 0.01,
                   "num_actions": num_act,
                   "replay_memory_size": mem_size,  # 200000 if using ER
                   "replay_init_size": 5000,
                   "batch_size": 32,
                   "fqi_reg_type": fqi_reg_type,  # "l2" or "prev"
                   }
    ## TTN
    hyper_sets_lstdq = OrderedDict([("nn_lr", np.power(10, [-3.0, -3.5, -4.0]).tolist()), 
                                    ("reg_A", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]),
                                    ("eps_decay_steps", [1]),
                                    ("update_freq", [1000]),  # default 1000
                                    ("data_length", [data_length]),
                                    ("fqi_rep", [fqi_rep]),
                                    # Data Augmentation Params
                                    ("data_aug_type", ['ras']),
                                    ("data_aug_prob", [0.0]),  #[0.0, 0.01, 0.05, 0.1]
                                    ("data_aug_loc", ['rep']), #['rep', 'fqi', 'both']
                                    ("random_shift_pad", [4]),
                                    ("ras_alpha", [0.9]), #0.6 , 0.8
                                    ("ras_beta", [1.1]),   #1.2 , 1.4
                                    # Input Transformations Params
                                    ("trans_type", ['none', 'sparse', 'sparse_random']),
                                    ("new_feat_dim", [32, 64, 128]),
                                    ("sparse_density", [0.1, 0.5])
                                    ])

    with open('Offline//Offline_Original_hyper_sets_lstdq.json', 'w') as fp:
        fp.write(json.dumps(hyper_sets_lstdq, indent=4))


    if alg in ("fqi"):
        hyper_sets = hyper_sets_lstdq
        TTN = True
    elif alg == "dqn":
        hyper_sets = hyper_sets_DQN
        TTN = False
    #
    hyperparams_all = list(itertools.product(*list(hyper_sets.values())))
    hyperparams = hyperparams_all

    # # Removing redundant experiments for data_aug_prob = 0
    hyperparams_filtered = []
    count = 0
    for i in range(len(hyperparams)):
        if(i%18 not in [1,2,3,4,5]):
            hyperparams_filtered.append(hyperparams[i])
            count += 1

    print(f"Number of Hyperparams: {count}")
    hyperparams = hyperparams_filtered

    # write to a file to maintain the indexes
    with open("Offline//Offline-sweep-parameters.txt", "w") as fp:
        for ix in range(len(hyperparams)):
            to_write = str(ix) + " " + '-'.join([str(x) for x in hyperparams[ix]])
            fp.write(to_write + "\n")
    fp.close()

    times = []
    start_time = time.perf_counter()

    prev_action_flag = 1
    num_repeats = num_rep  # 10

    hyper = hyperparams[hyper_num]
    
    #TODO: Add the following params too for sweep
    #TODO: num_updates_pre_train 
    #TODO: mem_size
    #TODO: type_of_augmentation (rep, fqi, both)
    #TODO: use traget_network 

    output_dir_path = "Offline//Training_{}_offline_env_{}_mem_size_{}_hyper_num_{}_date_{}".format(
                                                                                                 alg_type,
                                                                                                 en,
                                                                                                 mem_size,
                                                                                                 hyper_num,
                                                                                                 datetime.today().strftime("%d_%m_%Y")
                                                                                                )

    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    files_name = "{}//".format(output_dir_path)


    if rnd:
        files_name = files_name + '_rnd'

    if initial_batch:
        files_name = files_name + '_initialbatch_'

    if alg == 'fqi':
        log_file = files_name + str(alg)
    if alg == 'dqn':
        log_file = files_name + str(alg)


    data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])

    if alg in ('lstdq', 'fqi'):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("reg_A", hyper[1]),
                              ("eps_decay_steps", hyper[2]),
                              ("update_freq", hyper[3]),
                              ("data_length", hyper[4]),
                              ("fqi_rep", hyper[5]),
                              # Data Augmentation Params
                              ("data_aug_type", hyper[6]),
                              ("data_aug_prob", hyper[7]),
                              ("data_aug_loc", hyper[8]),
                              ("random_shift_pad", hyper[9]),
                              ("ras_alpha", hyper[10]), #0.6 , 0.8
                              ("ras_beta", hyper[11]),   #1.2 , 1.4
                              # Input Transformation Params
                              ("trans_type", hyper[12]),
                              ("new_feat_dim", hyper[13]),
                              ("sparse_density", hyper[14])
                              ])

        print(f"Parameters: {params}")
        # print(f"Nnet params: {nnet_params}")

    elif alg in ("dqn"):
        params = OrderedDict([("nn_lr", hyper[0]),
                              ("eps_decay_steps", hyper[1])])

    # saved_state_list_all = np.load(starting_state_path)  # np.load starting states

    # #######                        Creating Identity matrix                      #######
    if(params["trans_type"] == "none"):
        sparse_matrix = np.identity(input_dim)

    # #######                        Creating Sparse matrix                      #######
    elif(params["trans_type"] == "sparse"):
        sparse_matrix = np.random.choice(2, size=(input_dim, params["new_feat_dim"]), p=[1 - params["sparse_density"], params["sparse_density"]])
        #making sure that original features occur atleast once in the new features
        for row in range(sparse_matrix.shape[0]):
            sparse_matrix[row][row] = 1.0

    elif(params["trans_type"] == "sparse_random"):
        #######                        Creating Sparse random matrix                      #######
        sparse_matrix = np.array(sparse.random(input_dim, params["new_feat_dim"], density=params["sparse_density"], data_rvs=np.random.randn, dtype=np.float32).todense())
        sparse_matrix = (0.35 * sparse_matrix) / (np.sqrt(input_dim))
        #making sure that original features occur atleast once in the new features
        for row in range(sparse_matrix.shape[0]):
            sparse_matrix[row][row] = 1.0

    print(f"sparse_matrix: {sparse_matrix} , sparse_matrix.shape: {sparse_matrix.shape}")


    hyperparam_returns = []
    hyperparam_values = []
    hyperparam_episodes = []
    hyperparam_losses = []

    hyperparam_avgreturns = []
    hyperparam_avgvalues = []
    hyperparam_avgepisodes = []
    hyperparam_avgepisodevalues = []

    hyperparam_stdreturns = []
    hyperparam_stdvalues = []
    hyperparam_stdepisodes = []
    hyperparam_stdepisodevalues = []

    hyperparam_final_avgreturns = []
    hyperparam_final_avgvalues = []
    hyperparam_final_avgepisodes = []
    hyperparam_final_avgepisodevalues = []

    hyperparam_final_stdreturns = []
    hyperparam_final_stdvalues = []
    hyperparam_final_stdepisodes = []
    hyperparam_final_stdepisodevalues = []

    for rep in range(num_repeats):

        rand_seed = rep * 32  # 332

        if en != "catcher":
            env.seed(rand_seed)

        T.manual_seed(rand_seed)
        np.random.seed(rand_seed)

        with open(log_file + ".txt", 'a') as f:
            print("Start! Seed: {}".format(rand_seed), file=f)

        # saved_state_list = saved_state_list_all[rep * num_epi_per_itr:rep * num_epi_per_itr + num_epi_per_itr]

        #############################################
        start_run_time = time.perf_counter()

        if TTN:

            nn = TTNAgent_online_offline_mix(gamma, nnet_params=nnet_params, other_params=params, sparse_matrix=sparse_matrix,
                                             input_dims=input_dim, num_units_rep=128, dir=data_dir, offline=offline,
                                             num_tiling=16, num_tile=4, method_sarsa=method_sarsa,
                                             tilecoding=tilecoding, status=status)

        else:
            nn = DQNAgent(gamma, q_nnet_params, params, input_dims=input_dim, dir=data_dir, status=status)

        if initial_batch == False:

            if alg == 'dqn' or (TTN and nnet_params['replay_memory_size'] > 0):
                print("initialize buffer")
                frame_history = []
                prev_state = None
                prev_action = None
                done = 0
                if en != "catcher":
                    state_unnormal = env.reset()
                else:
                    state_unnormal = env.reset_game()

                state = process_state(state_unnormal)
                # populate the replay buffer
                while nn.memory.mem_cntr < nnet_params["replay_init_size"]:

                    if done:
                        if en != "catcher":
                            state_unnormal = env.reset()
                        else:
                            state_unnormal = env.reset_game()

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
                    prev_state = state
                    prev_action = action
                    state = new_state

        else:
            nn.memory.offline = True
            nn.memory.load_mem()

        prev_state = None
        prev_action = None

        run_returns = []
        run_vals = []
        run_episode_length = []
        run_avg_episode_values = []

        run_losses = []

        for itr in range(1):

            ## do offline-step update before running the agent
            ## Always pre-train in Offline
            if True or (not os.path.isfile("{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")):
                print("Pre-training")

                if TTN:
                    loss = nn.learn()

                else:  # DQN
                    loss = nn.learn()

                loss1 = loss
                diff_loss = 1
                t = 0

                # while t < num_updates_pretrain: # and diff_loss> epsilon_stop_training:
                #     for j in range(params["update_freq"] + 5):
                #         if TTN:
                #             loss = nn.learn()
                #
                #         else:  # DQN
                #             loss = nn.learn()
                #
                #     diff_loss = abs(loss-loss1)
                #     print(loss)
                #     print(diff_loss)
                #     loss1= loss
                #     t += 1

                # How do you control number of fqi updates in pre-train offline?

                # batch_size = 64
                for j in range(num_updates_pretrain): #num_updates_pretrain = num_epoch = 100
                    num_iteration_feature = int(mem_size / nnet_params['batch_size'])
                    shuffle_index = np.arange(nnet_params['replay_memory_size'])
                    np.random.shuffle(shuffle_index)

                    for itr in range(num_iteration_feature):

                        if TTN:
                            loss = nn.learn_nn_feature_fqi(itr, shuffle_index)
                            # print(loss)

                        else:
                            loss = nn.learn_nn_feature(itr, shuffle_index)
                            # print(loss)

                if TTN:
                    T.save(nn.q_eval.state_dict(), "{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")

                    T.save(nn.lin_weights, "{}//lin_weights_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")
                else:
                    T.save(nn.q_eval.state_dict(), "{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")

                    T.save(nn.q_next.state_dict(), "{}//feature_next_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")

            else:
                print("features and lin_weights already exist, skipping pretrianing!")
                if TTN:
                    nn.q_eval.load_state_dict(T.load("{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt"))
                    nn.lin_weights = T.load("{}//lin_weights_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt")
                else:
                    nn.q_eval.load_state_dict(T.load("{}//feature_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt"))
                    nn.q_next.load_state_dict(
                        T.load("{}//feature_next_{}_{}_{}_{}".format(output_dir_path, alg, en, mem_size, num_updates_pretrain) + ".pt"))
            ##
            ## end of offline step

            run_avgreturns = []
            run_avgvals = []
            run_avgepisode_length = []

            done = 0
            episodes = 0
            val = 0.0
            q_values_episode = 0
            ret = 0
            episode_length = 0
            count_10epi = 0

            if en != "catcher":
                env.reset()
            else:
                env.reset_game()
            # game = env.unwrapped
            # env.state = saved_state_list[count_10epi]
            # state_unnormal = env.state
            # state = process_state(state_unnormal)

            # state_unnormal = env.unwrapped.state
            if en != "catcher":
                state_unnormal = env.reset()
            else:
                state_unnormal = env.reset_game()

            state = process_state(state_unnormal)

            start_run_time = time.perf_counter()

            i = 0

            while episodes < num_epi_per_itr + 1:

                i += 1

                if done:
                    run_returns.append(ret)
                    run_vals.append(val)
                    run_episode_length.append(episode_length)
                    run_avg_episode_values.append((q_values_episode / episode_length))

                    run_avgreturns.append(ret)
                    run_avgvals.append(val)
                    run_avgepisode_length.append(episode_length)

                    print(episodes, i, round(val, 2), ret, "number episode from 10:", count_10epi,
                          "avegar over 10 episode:", round(np.mean(run_avgreturns), 3),
                          "avegare return across last 100 episodes:", round(np.mean(run_returns[-100:]), 3),
                          "state values:", (q_values_episode / episode_length), episode_length)

                    with open(log_file + ".txt", 'a') as f:
                        print(episodes, i, round(val, 2), ret, "number episode from 10:", count_10epi,
                          "avegar over 10 episode:", round(np.mean(run_avgreturns), 3),
                          "avegare return across last 100 episodes:", round(np.mean(run_returns[-100:]), 3),
                          "state values:", (q_values_episode / episode_length), episode_length, file=f)



                    count_10epi += 1
                    episode_length = 0
                    q_values_episode = 0
                    val = 0.0
                    ret = 0.0
                    i = 0
                    episodes += 1

                    if en != "catcher":
                        env.reset()
                    else:
                        env.reset_game()
                    # env.state = saved_state_list[count_10epi]
                    # state_unnormal = env.state
                    # state = process_state(state_unnormal)

                    # state_unnormal = env.unwrapped.state
                    if en != "catcher":
                        state_unnormal = env.reset()
                    else:
                        state_unnormal = env.reset_game()

                    state = process_state(state_unnormal)

                # Get action
                if TTN:
                    action = nn.choose_action(state)
                    q_values = nn.lin_values

                else:  # DQN
                    action, q_values = nn.choose_action(state)

                # run_values.append(T.mean(q_values.data, axis=1)[0].detach())
                q_values_episode += T.mean(q_values.data)

                # update parameters and add to the memory
                # .....

                # do one step in the environment
                episode_length += 1
                new_state_unnormal, reward, done, info = env.step(np.array(action))  # action is a 1x1 matrix
                new_state = process_state(new_state_unnormal)

                # update saved states
                prev_state = state
                prev_action = action
                state = new_state  # action is a 1x1 matrix
                val += reward
                ret += (gamma ** i) * reward

            ## store average values at each iteration
            hyperparam_avgreturns.append(np.mean(run_avgreturns))
            hyperparam_avgvalues.append(np.mean(run_avgvals))
            hyperparam_avgepisodes.append(np.mean(run_avgepisode_length))
            hyperparam_avgepisodevalues.append(np.mean(run_avg_episode_values))

            hyperparam_stdreturns.append(np.std(run_avgreturns))
            hyperparam_stdvalues.append(np.std(run_avgvals))
            hyperparam_stdepisodes.append(np.std(run_avgepisode_length))
            hyperparam_stdepisodevalues.append(np.std(run_avg_episode_values))

        hyperparam_returns.append(run_returns)
        hyperparam_values.append(run_avg_episode_values)
        hyperparam_episodes.append(run_episode_length)

        np.save(files_name + 'hyperparam_avgreturns', hyperparam_avgreturns)
        np.save(files_name + 'hyperparam_avgepisodes', hyperparam_avgepisodes)
        np.save(files_name + 'hyperparam_avgepisodevalues', hyperparam_avgepisodevalues)

        np.save(files_name + 'hyperparam_stdreturns', hyperparam_stdreturns)
        np.save(files_name + 'hyperparam_stdepisodes', hyperparam_stdepisodes)
        np.save(files_name + 'hyperparam_stdepisodevalues', hyperparam_stdepisodevalues)

        np.save(files_name + 'hyperparam_returns', hyperparam_returns)
        np.save(files_name + 'hyperparam_values', hyperparam_values)
        np.save(files_name + 'hyperparam_episodes', hyperparam_episodes)

        hyperparam_final_avgreturns.append(hyperparam_returns)
        hyperparam_final_avgvalues.append(hyperparam_values)
        hyperparam_final_avgepisodes.append(hyperparam_episodes)

    # np.save(files_name + 'hyperparam_final_avgreturns', np.mean(hyperparam_final_avgreturns, axis=0))
    # np.save(files_name + 'hyperparam_final_avgvalues', np.mean(hyperparam_final_avgvalues, axis=0))
    # np.save(files_name + 'hyperparam_final_avgepisodes', np.mean(hyperparam_final_avgepisodes, axis=0))
    #
    # np.save(files_name + 'hyperparam_final_stdreturns', np.std(hyperparam_final_avgreturns, axis=0))
    # np.save(files_name + 'hyperparam_final_stdvalues', np.std(hyperparam_final_avgvalues, axis=0))
    # np.save(files_name + 'hyperparam_final_stdepisodes', np.std(hyperparam_final_avgepisodes, axis=0))

    np.save(files_name + 'hyperparam_final_avgreturns', np.mean(hyperparam_returns, axis=0))
    np.save(files_name + 'hyperparam_final_avgvalues', np.mean(hyperparam_values, axis=0))
    np.save(files_name + 'hyperparam_final_avgepisodes', np.mean(hyperparam_episodes, axis=0))

    np.save(files_name + 'hyperparam_final_stdreturns', np.std(hyperparam_returns, axis=0))
    np.save(files_name + 'hyperparam_final_stdvalues', np.std(hyperparam_values, axis=0))
    np.save(files_name + 'hyperparam_final_stdepisodes', np.std(hyperparam_episodes, axis=0))



















