import gym
from envs.gridworld import GridHardRGBGoalAll
from utils.env_utils import process_state_constructor
from ple.games.catcher import Catcher
from ple import PLE

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

ple_rewards = {
        "positive": 1.0,
        "negative": -1.0,
        "tick": 0.0,
        "loss": -0.0,
        "win": 5.0
    }

def get_env(env_name, seed= 0):
    ## select environment
    
    if env_name == "Mountaincar":
        env = gym.make('MountainCar-v0')
        input_dim = env.observation_space.shape[0]
        env._max_episode_steps = 1000 
        num_act = 3 #TODO: These lines can be replaced with env.action_space.n
    elif env_name == "Acrobot":
        env = gym.make('Acrobot-v1')
        input_dim = env.observation_space.shape[0]
        num_act = 3
    elif env_name == "LunarLander":
        env = gym.make('LunarLander-v2')
        input_dim = env.observation_space.shape[0]
        num_act = 4
    elif env_name == "cartpole":
        env = gym.make('CartPole-v0')
        input_dim = env.observation_space.shape[0]
        num_act = 2
    elif env_name == 'gridhard':
        goal_id = 106 # position of the goal: this should be always fixed to this one to be consistent across experiments.
        env = GridHardRGBGoalAll(goal_id) # position of the goal: this should be always fixed to this one to be consistent across experiments.
        input_dim = (3, 15, 15)
        num_act = 4

    elif env_name == "catcher":
        game = Catcher(init_lives=1)
        process_state = process_state_constructor(env_name)
        env = PLE(game, fps=30, state_preprocessor=process_state, display_screen=False, reward_values=ple_rewards, rng=seed)
        input_dim = 4
        num_act = 3

    else:
        raise ValueError("Environment's name doesn't exist: {}".format(env_name, seed))
    
    return env, input_dim, num_act