##Testing
 ## OFFLINE-ONLINE REINFORCEMENT LEARNING: EXTENDING BATCH AND ONLINE RL


Code for Offline-Online Reinforcement Learning with a Two-Timescale Networks. 

Repo is setup for gym environments in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [PyTorch 1.7](https://github.com/pytorch/pytorch) and Python 3.7. 

### Overview

To generate a data-set from a good policy needs to be learned by running:
```
python create_data.py --hyper_num 15 --mem_size 1000 --num_step_ratio_mem 50000 --en 'Mountaincar'

```
This will save a generated buffer as the dataset by which the agent can train offline-online or offline or set that as the initial buffer for online. The agent can then train by running:
```
python create_data.py --offline_online_training 'offline_online' --tr_hyper_num 15 
```

The ```hyper_num``` assigns corresponding values to the hyper-parameters. For the Two-Timescale Networks, we used
two hyper-parameters, one for the learning rate and one for the regularizer coefficient. 

