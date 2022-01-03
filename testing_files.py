import numpy as np
import torch as T

file_path = "Data/Mountaincar_1000.npy"

d = np.load(file_path, allow_pickle=True)

state_memory = T.tensor((d.item().get('state')), dtype=T.float32)
new_state_memory = T.tensor((d.item().get('nstate')), dtype=T.float32)
action_memory = T.tensor((d.item().get('action')), dtype=T.int64)
reward_memory = T.tensor((d.item().get('reward')), dtype=T.float32)
terminal_memory = T.tensor((d.item().get('done')), dtype=T.bool)
mem_cntr = 1000

print(action_memory.shape)