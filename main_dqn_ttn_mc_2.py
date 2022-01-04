#import gym
import numpy as np
from dqn_agent import DQNAgent
from ttn_agent_online import TTNAgent_online
from ttn_agent_online_tc import TTNAgent_online_tc
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
def main(alg_type, hyper_num, fqi_length_num, mem_size, num_rep, ind, offline, fqi_rep_num, chunk_num, num_step_ratio_mem, num_epi_per_itr, num_updates, method_sarsa, envtest, feature, regularizer):

    fqi_lengths = [mem_size, 6000, 10000, 20000, 30000]
    fqi_length = fqi_lengths[fqi_length_num]

    fqi_reps = [1, 10, 50, 100, 300]
    fqi_rep = fqi_reps[fqi_rep_num]

    if len(sys.argv) > 4:
        gpu_index = sys.argv[4]
    else:
        gpu_index = ""  # ie. use cpu

    input_dim = 2  # state dimension

    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = gym.make('MountainCar-v0')
    rand_seed = num_rep * 32  # 332
    env.seed(rand_seed)
    T.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    if feature == 'tc':
        tilecoding = 1
    else:
        tilecoding = 0


    np.set_printoptions(linewidth=200, suppress=True, threshold=10000)


    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    ### Reward: -1 for each time step, until the goal position of 0.5 is reached.
    ### Action: 0 -> push left, 1-> no push, 2 -> push right
    ### Starting state: Random position from -0.6 to -0.4 with no velocity.

    def process_state(state, normalize=True):
        # states = np.array([state['position'], state['vel']])
        states = np.array([state[0], state[1]])
        if normalize:
            states[0] = (states[0] + 1.2) / (0.6 + 1.2)
            states[1] = (states[1] + 0.07) / (0.07 + 0.07)
        return states

    # test_states, _ = utils.catcher_test_data(image=False, normalize=True)


    # if len(sys.argv) > 1:
    #     alg = sys.argv[1]
    # else:
    #     alg = "fqi"
    #
    # if len(sys.argv) > 2:
    #     file_index = int(sys.argv[2])
    # else:
    #     file_index = 0
    #
    # if len(sys.argv) > 3:
    #     if sys.argv[3] == "v":
    #         image_based = True
    #     else:
    #         image_based = False

    alg = alg_type
    image_based = 0
    #dqn:


    if image_based:
        hyper_sets_DQN = OrderedDict([("nn_lr", np.power(10, [-4.00])),
                                      ("eps_decay_steps", [500000]),
                                     ])
    else:
        # For nonimage
        hyper_sets_DQN = OrderedDict([("nn_lr", np.power(10, [-3.25, -3.5, -3.75, -4.0, -4.25])),
                                      ("eps_decay_steps", [10000, 20000, 40000]),
                                     ])

    # hyper_sets_Q = OrderedDict([("nn_lr", np.power(10, [-3.75, -4.0])),
    #                             ("eps_decay_steps", [20000, 40000]),
    #                             ])

    image_history_len = 2
    num_steps = 10000000 if image_based else num_step_ratio_mem #200000
    nnet_params = {"loss_features": 'semi_MSTDE', #"next_reward_state", next_state, semi_MSTDE
                   "beta1": 0.0,
                   "beta2": 0.99,
                   "eps_init": 1.0,
                   "eps_final": 0.01,
                   "num_actions": 3,
                   "replay_memory_size": mem_size,  # 200000 if using ER
                   "replay_init_size": 5000,
                   "pretrain_rep_steps": 0,
                   "freeze_rep": False,  # If True, freezes the representation after 'pretrain_rep_steps' steps
                   "batch_size": 32,
                   "fqi_reg_type": regularizer,# "l2" or "prev"
                    }


    if image_based:
        hyper_sets_lstdq = OrderedDict([("nn_lr", np.power(10, [-5.0])),
                                        ("reg_A", [10**-3]),  # can also do reg towards previous weights
                                        ("eps_decay_steps", [1]),
                                        ("update_freq", [10000]),
                                        ("fqi_length", [fqi_length]),
                                        ("fqi_rep", [fqi_rep]),
                                        ("chunk_num", [chunk_num])
                                    ])
    else:
        hyper_sets_lstdq = OrderedDict([("nn_lr", np.power(10, [-2.0, -2.5, -3.0, -3.5, -4.0])),
                                        ("reg_A", np.power(10, [0.0, -1.0, -2.0, -3.0])),  # can also do reg towards previous weights
                                        ("eps_decay_steps", [1]),
                                        ("update_freq", [100]), #1000
                                        ("fqi_length", [fqi_length]),
                                        ("fqi_rep", [fqi_rep]),
                                        ("chunk_num", [chunk_num])
                                        ])

    # DQN
    q_nnet_v = {"replay_memory_size": mem_size,
                "replay_init_size": 50000,
                "update_target_net_steps": 10000,
                "update_fqi_steps": 500000}  # for levine (LS-DQN)

    q_nnet_nv = {"replay_memory_size": mem_size,
                 "replay_init_size": 5000,
                 "update_target_net_steps": 1000,
                 "update_fqi_steps": 50000}

    q_nnet_params = {"num_actions": 3,
                     "eps_init": 1.0,
                     "eps_final":  0.01,
                     "batch_size": 32,  # TODO try big batches
                     "replay_memory_size": mem_size, #20000,
                     "replay_init_size": 5000,
                     "update_target_net_steps": 1000
                    }
    if image_based:
        q_nnet_params.update(q_nnet_v)
    else:
        q_nnet_params.update(q_nnet_nv)


    checking_hyperparams = not image_based
    reward_shaping = False

    log_image_str = "v" if image_based else "nv"
    # log_file = "none"
    file_index = 1
    if alg == 'fqi':
        log_file = "Control_Online_env_{}_feature_{}_alg_{}_{}_data_{}_numupdates_{}_numepiperitr_{}_hypernum_{}_date_{}_rep_{}".format(
            envtest,
            feature, alg, method_sarsa, dir, num_updates, num_epi_per_itr, hyper_num, num_rep,
            datetime.today().strftime("%d_%m_%Y")
        )

    if alg == 'dqn':
        log_file = "Control_Online_env_{}_feature_{}_alg_{}_{}_data_{}_numupdates_{}_numepiperitr_{}_hypernum_{}_date_{}_rep_{}".format(
            envtest,
            feature, alg, method_sarsa, dir, num_updates, num_epi_per_itr, hyper_num, num_rep,
            datetime.today().strftime("%d_%m_%Y"))

    gamma = 0.99

    if alg in ("lstdq", "fqi"):
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



    all_results = []
    times = []
    last_weights = None

    start_time = time.perf_counter()
    index_run = 0

    # num_steps = 200000
    prev_action_flag = 1
    num_repeats = num_rep # 10
    # TTN = True

    saved_state_list_all = np.load(
        "..//MountainCar_test_states_random_dir15_numstates_2000.npy")  # np.load("MountainCar_test_states_3.npy")
    test_values_all = np.load("..//MountainCar_test_values_random_dir15_numstates_2000.npy")

    saved_state_list = saved_state_list_all[ind*num_epi_per_itr :ind*num_epi_per_itr + num_epi_per_itr]
    test_values = test_values_all[ind*num_epi_per_itr :ind*num_epi_per_itr + num_epi_per_itr]


    # for hyper in hyperparams:
    for iii in range(1):
        hyper = hyperparams[hyper_num]
        hyperparam_returns = []
        hyperparam_values = []
        hyperparam_episodes = []
        hyperparam_returns_avg10epi = []
        hyperparam_episode_length = []
        hyperparam_avg_episode_values = []

        data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])


        if alg in ('lstdq', 'fqi'):
            params = OrderedDict([("nn_lr", hyper[0]),
                                  ("reg_A", hyper[1]),
                                  ("eps_decay_steps", hyper[2]),
                                  ("update_freq", hyper[3]),
                                  ("fqi_length", hyper[4]),
                                  ("fqi_rep", hyper[5]),
                                  ("chunk_num", hyper[6])
                                  ])
            print(hyper[0], hyper[1], hyper[2])
            with open(log_file + ".txt", 'a') as f:
                for par in params:
                    if par in ("nn_lr", "reg_A", "eps_decay_steps", "update_freq", "fqi_length"):
                        print("hyper_parameter:" + par + ":{} ".format(params[par]), file=f)

        elif alg in ("dqn"):
            params = OrderedDict([("nn_lr", hyper[0]),
                                  ("eps_decay_steps", hyper[1])])
            with open(log_file + ".txt", 'a') as f:
                for par in params:
                    if par in ("nn_lr", "eps_decay_steps"):
                        print("hyper_parameter:" + par + ":{} ".format(params[par]), file=f)




        for rep in range(1):
            if checking_hyperparams:
                rng = file_index + rep  # reduce variance across different hyperparameters
            else:
                rng = file_index + rand_seed + rep

            with open(log_file + ".txt", 'a') as f:
                print("new run for the current setting".format(rep), file=f)

            if TTN:
                if tilecoding:
                    nn = TTNAgent_online_tc(gamma, nnet_params=nnet_params, other_params=params, input_dims=input_dim, num_units_rep=128, dir=None, offline=offline, num_tiling=16, num_tile=4, method_sarsa=method_sarsa)
                else:
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
                while nn.memory.mem_cntr < nnet_params["replay_init_size"]:
                    discount = gamma

                    if done:
                        # print("Game over", val, "\tStep", i)
                        # print('game')
                        val = 0.0
                        discount = 0.0
                        state_unnormal = env.reset()
                        state = process_state(state_unnormal)
                        if image_based:
                            frame_history = []

                    # get action
                    action = np.random.randint(0, 3)

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



            prev_state = None
            prev_action = None
            run_returns = []
            run_10returns = []
            run_avg10returns = []
            run_episodes = []
            run_values = []
            run_avg_episode_values = []
            run_losses = []
            run_episode_length = []
            done = 0
            episodes = 0
            val = 0.0
            q_values_episode = 0
            # state_unnormal = env.reset()
            # state = process_state(state_unnormal)

            episode_length = 0
            count_10epi = 0

            env.reset()
            game = env.unwrapped
            game.state = saved_state_list[count_10epi]
            state_unnormal = game.unwrapped.state

            # state_unnormal = env.reset()
            state = process_state(state_unnormal)

            # frame_history = []
            start_run_time = time.perf_counter()
            index_run += 1
            # print("Run", index_run)
            i=0
            while episodes< num_epi_per_itr-1:     #for i in range(num_steps):

                discount = gamma
                i +=1

                if done:

                    run_returns.append(val)
                    run_10returns.append(val)
                    run_episodes.append(i)
                    run_episode_length.append(episode_length)
                    run_avg_episode_values.append((q_values_episode / episode_length))

                    discount = 0.0
                    # episodes += 1
                    count_10epi += 1

                    print(episodes, i, round(val, 2), "number episode from 10:", count_10epi,
                          "avegar over 10 episode:", round(np.mean(run_10returns), 3),
                          "avegare return across last 100 episodes:", round(np.mean(run_returns[-100:]), 3),
                          "state values:", (q_values_episode / episode_length), episode_length)

                    with open(log_file + ".txt", 'a') as f:
                        print('time:', datetime.today().strftime("%H_%M_%S"), 'return', round(val, 2), i,
                              len(run_returns),
                              "\t{:.2f}".format(np.mean(run_returns[-100:])), 'number episode from 10:',
                              count_10epi, 'avegar over 10 episode:', round(np.mean(run_10returns)), "state values:",
                              (q_values_episode / episode_length), episode_length, file=f)

                    val = 0.0
                    episode_length = 0
                    q_values_episode = 0

                    # if TTN:
                    #     print(episodes, i, round(val, 2), round(np.mean(run_returns[-100:]), 3), nn.lin_values.data) #nn.lin_values.data
                    # else:
                    #     print(episodes, i, round(val, 2), round(np.mean(run_returns[-100:]), 3))
                    # avg_returns.append(np.mean(all_returns[-100:]))
                    val = 0.0
                    discount = 0.0
                    episodes += 1

                    env.reset()
                    game.state = saved_state_list[count_10epi]
                    state_unnormal = game.unwrapped.state
                    # state_unnormal = env.reset()
                    state = process_state(state_unnormal)

                    print(count_10epi)


                    if image_based:
                        frame_history = []

                # Get action
                if TTN:
                    action = nn.choose_action(state)
                    q_values = nn.lin_values

                else:  # DQN
                    action, q_values = nn.choose_action(state)

                run_values.append(T.mean(q_values.data, axis=1)[0].detach())
                q_values_episode += T.mean(q_values.data)

                # update parameters
                if prev_state is not None:
                    # prev_state_normal = process_state(prev_state)
                    # state_normal = process_state(state)
                    if prev_action_flag == 1:
                        nn.memory.store_transition_withnewaction(prev_state, np.squeeze(prev_action), reward, state, np.squeeze(action), int(done))
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
                reward = reward
                if reward> 1:
                    print(reward)
                val += reward


                if not checking_hyperparams and (i % 25000 == 0 and i > 0 or i == num_steps-1):  # temporary saves
                    np.save(log_file + '_returns' + str(index_run), run_returns)
                    #np.save(log_file + '_values' + str(index_run), run_values)
                    np.save(log_file + '_loss' + str(index_run), run_losses)
                    # print("Ret {}  Val {}  Loss {}".format(np.mean(run_returns[-100:]), np.mean(run_values[-1000:]), np.std(run_losses[-1000:])))

                    # saver.save(sess, "checkpoints/{}_{}_model.ckpt".format(alg, file_index), global_step=i)
                    with open(log_file + "_running" + str(index_run) + ".txt", "a") as f:
                        print("Ret {:.4f}".format(np.mean(run_returns[-100:])), "\tVal {:.4f}".format(np.mean(run_values[-1000:])),
                              "\tLoss {:.6f}".format(np.mean(run_losses[-1000:])), "\tTime {:.2f}".format((time.perf_counter() - start_time) / 60),
                              " Step {}".format(i//1000), file=f)

                # np.save("Returns_mc_original_num_step_{}_mem_size_{}_{}_{}_lenght_{}_date_{}_hyper_{}".format(num_step_ratio_mem, mem_size, log_image_str, alg, fqi_length, datetime.today().strftime("%d_%m_%Y"), hyper_num),
                #         run_returns)

            hyperparam_returns.append(run_returns)
            hyperparam_episodes.append(run_episodes)
            hyperparam_values.append(run_values)
            hyperparam_returns_avg10epi.append(run_avg10returns)
            hyperparam_episode_length.append(run_episode_length)
            hyperparam_avg_episode_values.append(run_avg_episode_values)

            # np.save("fqi_mc_hyperparam_returns"+str(fqi_length)+str(hyper_num)+str(mem_size), hyperparam_returns)
            # np.save("Returns_mc_num_step_{}_mem_size_{}_{}_{}_lenght_{}_date_{}_hyper_{}".format(num_step_ratio_mem, mem_size, log_image_str, alg, fqi_length,datetime.today().strftime("%d_%m_%Y"), hyper_num), hyperparam_returns)

            data['state'] = nn.memory.state_memory
            data['action'] = nn.memory.action_memory
            data['reward'] = nn.memory.reward_memory
            data['nstate'] = nn.memory.new_state_memory
            data['naction'] = nn.memory.new_action_memory
            data['done'] = nn.memory.terminal_memory
            # data = dict([('state', []), ('action', []), ('reward', []), ('nstate', []), ('naction', []), ('done', [])])

            np.save(
                "Control_Online_episodeLength_env_{}_feature_{}_alg_{}_{}_data_{}_numupdates_{}_numepiperitr_{}_hypernum_{}_date_{}_rep_{}_round_{}_3".format(
                    envtest,
                    feature, alg, method_sarsa, dir, num_updates, num_epi_per_itr, hyper_num, num_rep,
                    datetime.today().strftime("%d_%m_%Y"), rep)
                ,
                hyperparam_episode_length)

            np.save(
                "Control_Online_episodeValue_env_{}_feature_{}_alg_{}_{}_data_{}_numupdates_{}_numepiperitr_{}_hypernum_{}_date_{}_rep_{}_round_{}_3".format(
                    envtest,
                    feature, alg, method_sarsa, dir, num_updates, num_epi_per_itr, hyper_num, num_rep,
                    datetime.today().strftime("%d_%m_%Y"), rep),
                hyperparam_avg_episode_values)

        if checking_hyperparams:
            temp_res = {}
            temp_res.update(params)
            temp_res['log_name'] = log_file
            temp_res["returns"] = copy.deepcopy(hyperparam_returns)
            temp_res["state_values"] = copy.deepcopy(hyperparam_values)
            temp_res['episodes'] = copy.deepcopy(hyperparam_episodes)
            all_results.append(temp_res)

    if checking_hyperparams:
        np.save("res_{}.npy".format(log_file), all_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='fqi')
    parser.add_argument('--hyper_num', type=int, default=10)
    parser.add_argument('--fqi_length_num', type=int, default=0)
    parser.add_argument('--mem_size', type=int, default=50000)
    parser.add_argument('--num_rep', type=int, default=7)
    parser.add_argument('--ind', type=int, default=7)
    parser.add_argument('--offline', type=bool, default=False)
    parser.add_argument('--fqi_rep_num', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=1)
    parser.add_argument('--num_step_ratio_mem', type=int, default=90000)
    parser.add_argument('--num_epi_per_itr', type=int, default=100)
    parser.add_argument('--num_updates', type=int, default=1000)
    parser.add_argument('--method_sarsa', type=str, default='expected-sarsa')  # expected-sarsa, q-learning
    parser.add_argument('--envtest', type=str, default='mc')
    parser.add_argument('--feature', type=str, default='learned_fet')  # 'tc', 'learned_fet'
    parser.add_argument('--regularizer', type=str, default='l2')




    args = parser.parse_args()

    main(args.algo, args.hyper_num, args.fqi_length_num, args.mem_size, args.num_rep, args.ind, args.offline, args.fqi_rep_num,
         args.chunk_num, args.num_step_ratio_mem, args.num_epi_per_itr, args.num_updates,
         args.method_sarsa,
         args.envtest, args.feature, args.regularizer)





