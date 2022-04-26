import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# %matplotlib inline


# No Augmentation
offline_data_path = "Offline/data_augmentation_rep_learn/data_aug_type_ras_data_aug_prob_0.0_ras_alpha_0.6_ras_beta_1.2/"
offline_data = os.listdir(offline_data_path)

for data in offline_data:
    if("hyperparam_final_avgreturns" in data):
        offline_data_path_avg_ret = offline_data_path + data
        offline_data_avg_ret = np.load(offline_data_path_avg_ret).astype(np.float)
    elif("hyperparam_final_avgepisodes" in data):
        offline_data_path_avg_eps = offline_data_path + data
        offline_data_avg_eps = np.load(offline_data_path_avg_eps).astype(np.float)

mavg_offline_data_avg_eps = []
w = 10
for i in range(len(offline_data_avg_eps)):
    mavg_offline_data_avg_eps.append(np.mean(offline_data_avg_eps[i:i+w]))     
    

# Augmentation - 1
offline_data_aug_1_path = "Offline/data_augmentation_rep_learn/data_aug_type_ras_data_aug_prob_0.1_ras_alpha_0.6_ras_beta_1.2/"
offline_data_aug_1 = os.listdir(offline_data_aug_1_path)

for data in offline_data_aug_1:
    if("hyperparam_final_avgreturns" in data):
        offline_data_aug_1_path_avg_ret = offline_data_aug_1_path + data
        offline_data_aug_1_avg_ret = np.load(offline_data_aug_1_path_avg_ret).astype(np.float)
    elif("hyperparam_final_avgepisodes" in data):
        offline_data_aug_1_path_avg_eps = offline_data_aug_1_path + data
        offline_data_aug_1_avg_eps = np.load(offline_data_aug_1_path_avg_eps).astype(np.float)

#Moving average        
mavg_offline_data_aug_1_avg_eps = []
w = 10
for i in range(len(offline_data_avg_eps)):
    mavg_offline_data_aug_1_avg_eps.append(np.mean(offline_data_aug_1_avg_eps[i:i+w]))
        

# Augmentation - 2
offline_data_aug_2_path = "Offline/data_augmentation_rep_learn/data_aug_type_ras_data_aug_prob_0.1_ras_alpha_0.6_ras_beta_1.4/"
offline_data_aug_2 = os.listdir(offline_data_aug_2_path)

for data in offline_data_aug_2:
    if("hyperparam_final_avgreturns" in data):
        offline_data_aug_2_path_avg_ret = offline_data_aug_2_path + data
        offline_data_aug_2_avg_ret = np.load(offline_data_aug_2_path_avg_ret).astype(np.float)
    elif("hyperparam_final_avgepisodes" in data):
        offline_data_aug_2_path_avg_eps = offline_data_aug_2_path + data
        offline_data_aug_2_avg_eps = np.load(offline_data_aug_2_path_avg_eps).astype(np.float)

#Moving average        
mavg_offline_data_aug_2_avg_eps = []
w = 10
for i in range(len(offline_data_avg_eps)):
    mavg_offline_data_aug_2_avg_eps.append(np.mean(offline_data_aug_2_avg_eps[i:i+w]))
    
    
# Augmentation - 3
offline_data_aug_3_path = "Offline/data_augmentation_rep_learn/data_aug_type_ras_data_aug_prob_0.1_ras_alpha_0.8_ras_beta_1.2/"
offline_data_aug_3 = os.listdir(offline_data_aug_3_path)

for data in offline_data_aug_3:
    if("hyperparam_final_avgreturns" in data):
        offline_data_aug_3_path_avg_ret = offline_data_aug_3_path + data
        offline_data_aug_3_avg_ret = np.load(offline_data_aug_3_path_avg_ret).astype(np.float)
    elif("hyperparam_final_avgepisodes" in data):
        offline_data_aug_3_path_avg_eps = offline_data_aug_3_path + data
        offline_data_aug_3_avg_eps = np.load(offline_data_aug_3_path_avg_eps).astype(np.float)

#Moving average        
mavg_offline_data_aug_3_avg_eps = []
w = 10
for i in range(len(offline_data_avg_eps)):
    mavg_offline_data_aug_3_avg_eps.append(np.mean(offline_data_aug_3_avg_eps[i:i+w]))
    
    
# Augmentation - 4
offline_data_aug_4_path = "Offline/data_augmentation_rep_learn/data_aug_type_ras_data_aug_prob_0.1_ras_alpha_0.8_ras_beta_1.4/"
offline_data_aug_4 = os.listdir(offline_data_aug_4_path)

for data in offline_data_aug_1:
    if("hyperparam_final_avgreturns" in data):
        offline_data_aug_4_path_avg_ret = offline_data_aug_4_path + data
        offline_data_aug_4_avg_ret = np.load(offline_data_aug_4_path_avg_ret).astype(np.float)
    elif("hyperparam_final_avgepisodes" in data):
        offline_data_aug_4_path_avg_eps = offline_data_aug_4_path + data
        offline_data_aug_4_avg_eps = np.load(offline_data_aug_4_path_avg_eps).astype(np.float)

#Moving average        
mavg_offline_data_aug_4_avg_eps = []
w = 10
for i in range(len(offline_data_avg_eps)):
    mavg_offline_data_aug_4_avg_eps.append(np.mean(offline_data_aug_4_avg_eps[i:i+w]))
    

    
    
# plots
# sns.set(rc={'text.usetex': True})
plt.plot(mavg_offline_data_avg_eps, color='cyan', label='without-aug')
plt.plot(mavg_offline_data_aug_1_avg_eps, label='alpha:0.6, Beta:1.2, prob:0.1')
plt.plot(mavg_offline_data_aug_2_avg_eps, label='alpha:0.6, Beta:1.4, prob:0.1')
plt.plot(mavg_offline_data_aug_3_avg_eps, label='alpha:0.8, Beta:1.2, prob:0.1')
plt.plot(mavg_offline_data_aug_4_avg_eps, label='alpha:0.8, Beta:1.4, prob:0.1')
plt.legend()
plt.title(f"Offline Exp. Augmentation vs No Augmentation")
plt.ylabel(f"Avg. Episode Length")
plt.xlabel("No. of Episodes")
# sns.set(rc={'text.usetex': True})
plt.savefig("Offline Exp. Augmentation vs No Augmentation !!!!!!!!!!!!!!.png", dpi=300)
# plt.show()