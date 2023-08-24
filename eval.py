#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:



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
import os
os.environ['PYTHONHASHSEED']=str(seed)

from agent.all2 import *


# In[3]:


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
state_dim = 16 #number of feature + edge index

num_agents = 6
action_dim = 2
global_state_dim = (state_dim * num_agents) + num_agents 






state_dim = 10 + 12 #number of feature + edge index
old_state_dim = 10 + 6
num_agents = 12
action_dim = 2
global_state_dim = (old_state_dim * num_agents) + num_agents
old_global_state_dim = (16 * 6) + 6 
print(global_state_dim)
print(old_global_state_dim
     )




    
class Model(nn.Module):
    def __init__(self, envs, in_layer = 64):
        super().__init__()
        """
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, in_layer)),
            nn.Tanh(),
        )"""
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
        self.multihead_attention = nn.MultiheadAttention(state_dim, 2)
        self.conv_layers3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size =(4,10), stride=1, padding = 2),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6,7), stride=1, padding = 2),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size =(3,8), stride=1, padding = 2),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6,7), stride=1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 8), padding=(0, 2)) , # Adjust kernel_size and padding
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 8), padding=(0, 2)),
        )
        #self.norm1 = nn.LayerNorm(state_dim)
        self.multihead_attention2 = nn.TransformerEncoderLayer(d_model=old_state_dim, nhead=2,dim_feedforward = 128, batch_first  = True )
        self.multihead_attention3 = nn.TransformerEncoderLayer(d_model=old_state_dim, nhead=2,dim_feedforward = 128, batch_first  = True )
        self.multihead_attention4 = nn.TransformerEncoderLayer(d_model=state_dim, nhead=2,dim_feedforward = 256, batch_first  = True )
    def get_states(self, x,a= False):
        #out = self.multihead_attention(x, x, x,)
        x = self.multihead_attention4(x)
        #x = 0
        if a:
            x= x.unsqueeze(1)
        else : x= x.unsqueeze(0)
        #print(x.shape)
        x = self.conv_layers(x)
        if a :
            x = x.squeeze(1)
        else : x = x.squeeze(0)
        #x = self.multihead_attention2(x)
        return x

    def get_value(self, x):
        #out,_ = self.get_states(x)
        #global_state = torch.cat((x.clone().view(-1),torch.tensor(acts)),0,)
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
   
    xml_name = f"xml/SAMAPPO_my_reward_high_last{ver}_test.xml"
    #PATH = "Models/SAMAPPO12/Best_PPO_lstm_layers_120_1692195437_in_systemWaitingTime"
    #PATH2 = "Models/SAMAPPO12/Best_PPO_lstm_layers_120_1692200289_in_systemWaitingTime"
    #PATH3 = "Models/SAMAPPO12/Best_PPO_lstm_layers_120_1692207389_in_rewards"
    #p = "Models/SAMAPPO12/best"
    #last = "Models/SAMAPPO12/LAST_Best_PPO_lstm_layers_120_1692277577_in_systemWaitingTime"
    #lastone = "Models/SAMAPPO12/LAST_Best_PPO_lstm_layers_120_1692284338_in_systemWaitingTime"
    ll = "Models/SAMAPPO12/PY_LAST_Best_PPO_lstm_layers_120_1692295312_in_systemWaitingTime"
    ll2 = "Models/SAMAPPO12/PY_LAST_Best_PPO_lstm_layers_120_1692296805_in_rewards"
    ll3 = "Models/SAMAPPO12/PY_LAST_Best_PPO_lstm_layers_120_1692297958_in_systemWaitingTime"
    last = "Models/SAMAPPO12/last/best2"
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
    agent.load_state_dict(torch.load(last))
    #agent.critic.load_state_dict(b.critic.state_dict())
    #agent.multihead_attention2.load_state_dict(d.multihead_attention2.state_dict())
    #agent.actor.load_state_dict(b.actor.state_dict())
    #agent.multihead_attention4.load_state_dict(c.multihead_attention4.state_dict())
    #agent.conv_layers3.load_state_dict(c.conv_layers3.state_dict())
    #agent.critic.requires_grad_(False)
    #agent.multihead_attention2.requires_grad_(False)
    #agent.actor.requires_grad_(False)
    agent.eval()
    learning_rate = 1e-4# good results were acheived with 1e-3lr with no rewards normalisation
    update_epochs = 20
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    #assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    
    clip_coef=0.1

    # ALGO Logic: Storage setup
    num_envs = 1 
    
    update_freq = 900
    batch_size = update_freq * num_envs
    minibatch_size = 60 #//10
    #print(envs.observation_space.shape)
    #print(envs.action_space)
    
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

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    ent_coef = 0.01
    vf_coef = 0.1
    max_grad_norm = 0.5
    norm_adv = True
    target_kl = None
    clip_vloss = True
    anneal_lr = True
    start_time = time.time()
    total_timesteps = 20000
    gamma= 0.95
    gae_lambda = 0.9 
    #next_obs = torch.Tensor(envs.reset()[0]).to(device)
    #next_done = torch.zeros(num_envs).to(device)
 
    num_updates = total_timesteps #// batch_size
    episodes = 1000000
    up = 0
    obs = envs.reset()
    best_rewards =  float('-inf')
    best_agent =  float('inf')
    best_system =  float('inf')
    #for for episode in range(1, episodes + 1):
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
    
    
            #actss[tls] = torch.zeros_like(actions2).to(device)
            #probss[tls] = torch.zeros_like(logprobs2).to(device)
            n_actss[tls] = 0
    next_obs = torch.Tensor(torch.Tensor(list(envs.reset().values()))).to(device)

    while step < 900:
        #print(env.action_space.sample())
        #print(next_obs)
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
        i=0
        
        #print(action)
        #print(tls)
        #for a in action:
         #   print(i)
         #   tls_a[tls[i]] = a.cpu().numpy()
         #   i+=1
        
        
        #print(tls_a)
        i=0
        next_obs, reward, dones, info = envs.step(n_actss)
        #print(next_obs)
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
            print(f'mean_waiting_time is {mean_waiting_time / 900}')
            print(f'agent_waiting_time is {agent_waiting_time}')
            print(f'mean_speed is {mean_speed / 900}')
            print(f'system_total_stopped is {system_total_stopped}')

    envs.close()

"""
# In[13]:


#0.6 12 high
#sa
558.1583663115969
98077.45414862898
mean_waiting_time is 0.620175962568441
agent_waiting_time is 98077.45414862898
mean_speed is 8.363048521774884
system_total_stopped is 19878

#high 0.8 12
#sa
376.2415435541688
59192.12222222228
mean_waiting_time is 0.418046159504632
agent_waiting_time is 59192.12222222228
mean_speed is 8.984777225165725
system_total_stopped is 10862

#fixed

3779.2161970510215
311847.5345238094
mean_waiting_time is 4.199129107834469
agent_waiting_time is 311847.5345238094
mean_speed is 7.230660313573269
system_total_stopped is 38090


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.DataFrame({'waiting time': wait_list})

# Save the DataFrame to a CSV file
df.to_csv(r'vis\SAMAPPO\12\waiting_high40.6.csv', index=False)
plt.plot(x_steps, wait_list, label='system waiting time')


plt.title('Simulation  Progress')
plt.xlabel('Step')
plt.ylabel('waiting time')

# Add a legend
plt.legend()

# Display the plot
plt.show()


# In[16]:


if 1 :
    ver = '0.6'
    xml_name = f"xml/fixed_my_reward_high_12_{ver}.xml"
    envs = SumoEnvironment(net_file=f'12inter{ver}/12inter.net.xml',
                  route_file=f'12inter{0.8}/rou.rou.xml',delta_time = 4,output_file =xml_name,
                  use_gui=False,fixed_ts = True
                )
    update_freq = 720
    num_envs = 4

    b = envs.reset()
    #print(torch.Tensor(list(obs.values())))
    done = True
    step = 0
    print(envs.observation_space.shape)
    
    mean_waiting_time =0
    agent_waiting_time =0
    mean_speed = 0
    system_total_stopped = 0
    tls_a = {} 
    tls =list(obs.keys())
    sim_step = 0
    lst=[]
    wait_list = []
    speed_list = []
    x_steps = []

    
    while step < 900:
        #print(env.action_space.sample())
        if step%90 ==0:
            print(step)
        
        next_obs, reward, done, info = envs.step(None)#action.cpu().numpy())
        sim_step+=4
        envs.sumo.simulationStep(sim_step)
        #print(len(next_obs))
        mean_waiting_time +=  info["system_mean_waiting_time"]
        lst.append(mean_waiting_time)
        agent_waiting_time += info["agents_total_accumulated_waiting_time"]
        #print(len(next_obs))
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
            print(f'mean_waiting_time is {mean_waiting_time / 900}')
            print(f'agent_waiting_time is {agent_waiting_time}')
            print(f'mean_speed is {mean_speed / 900}')
            print(f'system_total_stopped is {system_total_stopped}')
            
    envs.close()


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.DataFrame({'waiting time': wait_list})

# Save the DataFrame to a CSV file
df.to_csv(r'vis\fixed\12\waiting_high4_0.6.csv', index=False)
plt.plot(x_steps, wait_list, label='system waiting time')


plt.title('Simulation  Progress')
plt.xlabel('Step')
plt.ylabel('waiting time')

# Add a legend
plt.legend()

# Display the plot
plt.show()


# In[ ]:




"""