
from agent.utils import *


import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
seed = 42
torch_deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
torch.use_deterministic_algorithms(mode=True)





def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer






state_dim = 10 + 6 #number of feature + edge index
num_agents = 6
action_dim = 2
global_state_dim = (state_dim * num_agents) + num_agents 




    
class Model(nn.Module):
    def __init__(self, envs, in_layer = 64):
        super().__init__()
        self.encoder_c = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, old_global_state_dim)),
            #nn.Tanh(),
            #layer_init(nn.Linear(in_layer, 1), std=1.0),
        ) 
        
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(old_global_state_dim, in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(old_state_dim,in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, envs.action_space.n), std=0.01),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size =(3,8), stride=1, padding = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6,7), stride=1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.multihead_attention = nn.TransformerEncoderLayer(d_model=state_dim, nhead=2,dim_feedforward = 256, batch_first  = True )
    def get_states(self, x,update= False):
        x = self.multihead_attention(x)
        if update:
            x= x.unsqueeze(1)
        else : x= x.unsqueeze(0)
        x = self.conv_layers(x)
        if a :
            x = x.squeeze(1)
        else : x = x.squeeze(0)
        return x

    def get_value(self, x):
        out = self.encoder_c(x)
        
        return self.critic(out)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        #print(logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env setup
num_steps = 3600
in_layer = 120






# In[10]:


if 1 :
    ver = '0.6'
   
    xml_name = f"xml/SAMAPPO.xml"
    path  = "path_to_saved_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    num_steps = 3600
    in_layer = 120
    envs = SumoEnvironment(net_file=f'12inter{ver}/12inter.net.xml',
                  route_file=f'12inter{ver}/rou.rou.xml',sumo_seed = seed,
                  use_gui=True,
                 min_green = 4, max_green=60,delta_time = 4, output_file =xml_name,
                  num_seconds=3600, reward_fn="my_reward")
    in_layer = 120
    k = 0
 

    agent = Model(envs,in_layer).to(device)
    agent.load_state_dict(torch.load(path))
    agent.eval()
    learning_rate = 1e-4# good results were acheived with 1e-3lr with no rewards normalisation
    update_epochs = 20
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    
    clip_coef=0.1

    # ALGO Logic: Storage setup
    num_envs = 1 
    
    update_freq = 900
    
    obs = torch.zeros((update_freq, num_agents,state_dim)).to(device)
    global_obs = torch.zeros((update_freq,1,global_state_dim)).to(device)
    
    actions = torch.zeros((update_freq, num_agents)).to(device)
    actions2 = torch.zeros((update_freq)).to(device)
    actss = {}
    n_actss = {}
    logprobs = torch.zeros((update_freq, num_agents)).to(device)
    logprobs2 = torch.zeros((update_freq)).to(device)
    probss = {}
    rewards = torch.zeros((update_freq)).to(device)
    dones = torch.zeros((update_freq)).to(device)
    values = torch.zeros((update_freq)).to(device)

    global_step = 0
    up = 0
    obs = envs.reset()
    best_rewards =  float('-inf')
    best_agent =  float('inf')
    best_system =  float('inf')
    infos2 = {}
    global_ob = 0
    mean_waiting_time =0
    agent_waiting_time =0
    mean_speed = 0
    system_total_stopped = 0
    step = 0
    tls_a = {} 
    tls =list(obs.keys())
    lst = []
    wait_list = []
    speed_list = []
    x_steps = []
    
    for tls in envs.ts_ids:
            n_actss[tls] = 0
    next_obs = torch.Tensor(torch.Tensor(list(envs.reset().values()))).to(device)

    while step < 900:
        if step%90 ==0:
            print(step)
        
        with torch.no_grad():
            x = agent.get_states(next_obs)
            i=0 
            acts = []
            probs = []
            for tls in envs.ts_ids:
                action, logprob, _,  = agent.get_action(x[i])
                acts.append(action)
                probs.append(logprob)
                i+=1
                #actss[tls][up] = action
                #probss[tls][up] = logprob
                n_actss[tls] = action
            
        actions[step] = action
        next_obs, reward, dones, info = envs.step(n_actss)
        next_obs = torch.Tensor(torch.Tensor(np.array(list(next_obs.values()))))
        mean_waiting_time +=  info["system_mean_waiting_time"]
        lst.append(mean_waiting_time)
        agent_waiting_time += info["agents_total_accumulated_waiting_time"]
        mean_speed +=  info["system_mean_speed"] 
        system_total_stopped += info["system_total_stopped"]
        wait_list.append(info["system_mean_waiting_time"])
        speed_list.append(info["system_mean_speed"])
        x_steps.append(step * 4)
        #print(len(next_obs))
        #done = t
        step+=1
        if step == 900:
            print(mean_waiting_time)
            print(agent_waiting_time)
            print(f'mean_waiting_time is {mean_waiting_time}')
            print(f'agent_waiting_time is {agent_waiting_time}')
            print(f'mean_speed is {mean_speed / 900}')
            print(f'system_total_stopped is {system_total_stopped}')

    envs.close()

