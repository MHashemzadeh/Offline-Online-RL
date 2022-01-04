
 ## OFFLINE-ONLINE REINFORCEMENT LEARNING: EXTENDING BATCH AND ONLINE RL


Code for Offline-Online Reinforcement Learning with a Two-Timescale Networks. 

This branch is setup for MinAtar (https://github.com/kenjyoung/MinAtar). It has five environments and for now, we are running all of these environments. You may want to start from "asterix" as the first environment. To run the experiments you need to install MinAtar by following its instruction. 


### Overview
The same as before, you need to run ```creat_data.py``` with one of the three settings: "online", "offline", "offline_online" to get the results. 

Right now, a dataset generated with a good policy is not available for these environments. As soon as it is created, it would be shared with you.

For these environemnts, we have '--tr_NUM_FRAMES'= 5000000, '--tr_TRAINING_FREQ' = 4 which you do not need to change them. To run for different random seeds, you may want to change the value of '--SEED'. The same as before, you need to change the ```hyper_num``` for each environments. 

```
python creat_data.py --offline_online_training 'offline_online' --tr_hyper_num 32 
```

The ```hyper_num``` assigns corresponding values to the hyper-parameters. For the Two-Timescale Networks, we used
two hyper-parameters, one for the learning rate and one for the regularizer coefficient. 


