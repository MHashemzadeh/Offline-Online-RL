import pickle
import numpy as np

from utils.tiles3 import *

class TCAgent():
    def __init__(self, params):

        self.dynamic = False
        self.full = False

        # control settings
        self.num_action = 3 # params.environment_params.num_action
        self.obs_dim = 4 #params.environment_params.obs_dim
        self.obs_limits = [[0,1,1],[0,1,1]] #params.environment_params.obs_limits

        # tiling settings
        self.num_tiles = 10 #params.feature_constructor_params.num_tiles
        self.num_tilings = 5  #params.feature_constructor_params.num_tilings
        self.use_bias = True #params.feature_constructor_params.use_bias
        self.feature_dim = 1024 #params.feature_constructor_params.feature_dim
        if self.use_bias:
            self.feature_dim = +1

        # self.tile_independently = params.feature_constructor_params.tile_independently
        # self.input_action = params.feature_constructor_params.input_action
        self.normalized = True #params.feature_constructor_params.normalized

        self.real_mode = False
        if self.use_bias:
            self.sparse_feature_size = self.num_tilings+1
        else:
            self.sparse_feature_size = self.num_tilings
        if self.normalized:
            self.feature_value = 1.0/np.sqrt(self.sparse_feature_size)
        else:
            self.feature_value = 1.0

        self.np_random = 1 #params.np_random

        self.scaled_obs = []
        for i in range(self.obs_dim):
            self.scaled_obs.append(0.0)

        if self.use_bias:
            self.iht = IHT(self.feature_dim-1)
        else:
            self.iht = IHT(self.feature_dim)

    def save_feature_constructor(self, path):
        with open(path+'.pkl', 'wb') as f:
            pickle.dump(self.iht, f)

    def get_features_sparse(self,current_state,features):
        for i in range(self.obs_dim):
            # self.scaled_obs[i] = current_state[i]*self.num_tiles
            self.scaled_obs[i] = ((current_state[i]-self.obs_limits[i][0])/self.obs_limits[i][2])*self.num_tiles
        if self.use_bias:
            features[:-1] = tiles(self.iht, self.num_tilings, self.scaled_obs)
            features[-1] = self.feature_dim-1
        else:
            features[:] = tiles(self.iht, self.num_tilings, self.scaled_obs)
        self.full = self.iht.full

    def load_feature_constructor(self, path):
        with open(path+'.pkl', 'rb') as f:
            self.iht = pickle.load(f)

    def feature_length(self, features):
        return self.sparse_feature_size


def init(params):
    return TCAgent(params)

def get_params():
    return ["feature_dim","num_tiles","num_tilings"]

if __name__ == "__main__":
    tc = TCAgent(None)
