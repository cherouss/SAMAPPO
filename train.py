#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import os


# In[4]:




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

from agent.utils import *





# In[6]:


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
state_dim = 16 #number of feature + edge index
num_agents = 6
action_dim = 2
global_state_dim = (state_dim * num_agents) + num_agents 

class Model2(nn.Module):
    def __init__(self,envs, in_layer = 64):
        super().__init__()
        """
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, in_layer)),
            nn.Tanh(),
        )"""
        
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim,in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, in_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(in_layer, 2), std=0.01),
        )
        self.multihead_attention = nn.MultiheadAttention(state_dim, 2)
        #self.norm1 = nn.LayerNorm(state_dim)
        self.multihead_attention2 = nn.TransformerEncoderLayer(d_model=state_dim, nhead=2,dim_feedforward = 128,dropout = 0.1, batch_first  = True )
    def get_states(self, x):
        #out = self.multihead_attention(x, x, x,)
        out = self.multihead_attention2(x)
        return out

    def get_value(self, x):
        #out,_ = self.get_states(x)
        #global_state = torch.cat((x.clone().view(-1),torch.tensor(acts)),0,)
        
        return self.critic(x)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        #print(logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
    
PATH = "Models/SAMAPPO/Best_PPO_lstm_layers_120_1692103769_in_systemWaitingTime"
envs = 'p'
b = Model2(envs,in_layer = 120)
b.load_state_dict(torch.load(PATH))
state_dim = 10 + 12 #number of feature + edge index
old_state_dim = 10 + 6
num_agents = 12
action_dim = 2
global_state_dim = (old_state_dim * num_agents) + num_agents
old_global_state_dim = (16 * 6) + 6 
print(global_state_dim)
print(old_global_state_dim
     )


# In[7]:








    
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





#args = parse_args()
run_name = f"{int(time.time())}"
track = False
name = f'SA_PPO_SUMO{run_name}'

writer = SummaryWriter(f"runs/new/12intersections_transfer_pyFile_{name}")





# TRY NOT TO MODIFY: seeding


#torch.backends.cudnn.deterministic = torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# env setup
num_steps = 3600
in_layer = 120
#envs = Agent(act_dim=action_dim,state_dim=state_dim)



envs = SumoEnvironment(net_file='data/6intersections/nromal/6inter.net.xml',
              route_file='data/6intersections/nromal/rou.rou.xml',
              use_gui=False,sumo_seed = seed,
             min_green = 4, max_green=60,delta_time = 4,begin_time = 0,
              num_seconds=3600, reward_fn="my_reward")

agent = Model(envs,in_layer).to(device)
agent.critic.load_state_dict(b.critic.state_dict())
#agent.multihead_attention2.load_state_dict(d.multihead_attention2.state_dict())
agent.actor.load_state_dict(b.actor.state_dict())
#agent.multihead_attention4.load_state_dict(c.multihead_attention4.state_dict())
#agent.conv_layers3.load_state_dict(c.conv_layers3.state_dict())
agent.critic.requires_grad_(False)
agent.multihead_attention2.requires_grad_(False)
agent.actor.requires_grad_(False)

learning_rate = 1e-3# good results were acheived with 1e-3lr with no rewards normalisation
update_epochs = 20
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
#assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"


clip_coef=0.1

num_envs = 1 

update_freq = 900
batch_size = update_freq * num_envs
minibatch_size = 90 #//10

obs = torch.zeros((update_freq, num_agents,state_dim)).to(device)
obs2 = torch.zeros((update_freq, num_agents,old_state_dim)).to(device)
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
ent_coef = 0.01
vf_coef = 0.1
max_grad_norm = 0.5
norm_adv = True
target_kl = None
clip_vloss = True
anneal_lr = True
start_time = time.time()
total_timesteps = 20000
gamma= 0.98
gae_lambda = 0.9 
num_updates = total_timesteps #// batch_size
episodes = 1000000
up = 0
best_rewards =  float('-inf')
best_agent =  float('inf')
best_system =  float('inf')
#for for episode in range(1, episodes + 1):
infos2 = {}
global_ob = 0
mean_waiting_time =0
agent_waiting_time =0
print(envs.ts_ids)
for update in range(1, num_updates + 1):
    aa = True
    next_obs = torch.zeros((num_agents,state_dim)).to(device)
    next_done = torch.zeros((1)).to(device)
    #print(next_obs.shape)
    envs.reset()
    
    
    for tls in envs.ts_ids:
        #print(envs.trafficlight.getAllProgramLogics(tls))
        actss[tls] = torch.zeros_like(actions2).to(device)
        probss[tls] = torch.zeros_like(logprobs2).to(device)
        n_actss[tls] = 0
        
    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, 900):
        #if up %500 ==0: up 
        global_step += 1 * num_envs
        #next_obs = torch.Tensor(next_obs)
        #print("next_obs.shape")
        #print(next_obs.shape)
        x = agent.get_states(next_obs)
        if aa:
            print(x.shape)
            aa= False
        obs[up] = next_obs
        obs2[up] = x
        dones[up] = next_done
        acts = []
        probs = []
        with torch.no_grad():
            i=0 
            for tls in envs.ts_ids:
                #print(x[i])
                action, logprob, _,  = agent.get_action(x[i])
                acts.append(action)
                probs.append(logprob)
                i+=1
                actss[tls][up] = action
                probss[tls][up] = logprob
                n_actss[tls] = action
        
                
                #values[up] = value.flatten()
        #print(acts)
        act_dict = {}
        #j= 0
        #for tls in envs.ts_ids:
         #   act_dict[tls] = acts[j]
          #  j+=1
            
        actions[up] = torch.Tensor(acts)
        #print(actss)
        #print(actions[up])
        #logprobs[up] = s[up] = value.flatten()
        logprobs[up] = torch.Tensor(probs)
        global_ob =  torch.cat((x.clone().view(-1),torch.tensor(acts)),0,)
        global_obs[up] =  global_ob
        values[up] = agent.get_value(global_ob)
        #print(envs)
        
        
            

        # TRY NOT TO MODIFY: execute the game and log data.
        #print(envs.step(action.cpu().numpy()))
        #print(actions[up])
        next_obs, reward, done, info = envs.step(n_actss)
        
        next_obs = np.array(list(next_obs.values()))
        
        reward_ = np.array(list(reward.values())).sum()
        done = any(list(done.values()))
        

        mean_waiting_time +=  info["system_mean_waiting_time"]
        agent_waiting_time += info["agents_total_accumulated_waiting_time"]

        rewards[up] = torch.tensor(reward_).to(device)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
        up+=1
        #if done:
         #   print(up)
            #envs.close()
            #envs.reset()
        
        if up == update_freq:
            infos2 = info
            print(infos2['agents_total_accumulated_waiting_time'])
            #print(obs2[150,2])
            up = 0
        
    with torch.no_grad():
            #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
                next_value = agent.get_value(global_ob)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(update_freq)):
                    if t == update_freq - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                # flatten the batch
            
                b_obs= obs.reshape((-1,) + (num_agents,state_dim))
                b_obs2 = obs2.reshape((-1,) + (num_agents,old_state_dim))
                #print(b_obs2)
                b_dones = dones.reshape(-1)
            
            
                b_logprobs = logprobs.reshape(-1,num_agents)
                b_logprobs2 = probss
                #print(b_logprobs.shape)
                b_actions = actions.reshape((-1,num_agents))
                b_actions2 = actss
                #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                b_advantages = advantages.reshape(-1)
                #print(b_advantages.shape)
                global_b_obs = global_obs.reshape((-1,global_state_dim))
                #print(global_obs.shape)
                #print(global_b_obs.shape)
                #print(returns)

                b_returns = returns.reshape(-1)
                #b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

                b_values = values.reshape(-1)
                #b_values = (b_values - b_values.mean()) / (b_values.std() + 1e-8)

                #print(b_obs)
                #print(b_returns)

                # Optimizing the policy and value network
                b_inds = np.arange(batch_size)
                clipfracs = [] 
    for epoch in range(update_epochs):

                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    logratio = torch.zeros_like(b_logprobs).to(device)
                    newvalue = agent.get_value(global_b_obs[mb_inds])
                    newvalue = newvalue.view(-1)
                    #newvalue = (newvalue - newvalue.mean()) / (newvalue.std() + 1e-8)
                    loss = 0
                    i = 0
                    x = agent.get_states(b_obs[mb_inds],True)
                    #x = b_obs[mb_inds]
                    #print(x.shape)
                    for tls in envs.ts_ids:
                        #print('obv all ')
                        #print(b_obs[mb_inds].shape)
                        #print('obv i ')
                        #print(b_obs[mb_inds][i].shape)
                        #print('obv :')
                        #print(b_obs[mb_inds,i].shape)
                        __, newlogprob, entropy = agent.get_action(x[:,i], b_actions2[tls][mb_inds])
                        #print(newlogprob.shape)
                        logratio[mb_inds,i] = (newlogprob - b_logprobs2[tls][mb_inds])
                        i+=1
                    #print(b_logprobs)
                        #logratio_ = logratio[mb_inds][i].view(-1)
                        #print(logratio_.shape)
                        #print(logratio)
                        #print(logratio.shape)
                        #print(b_returns[mb_inds])
                    ratio = logratio[mb_inds].exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio[mb_inds]).mean()
                        approx_kl = ((ratio - 1) - logratio[mb_inds]).mean()
                        clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                    ratio_ = logratio[mb_inds].exp().mean(1)
                    pg_loss1 = -mb_advantages * ratio_
                    pg_loss2 = -mb_advantages * torch.clamp(ratio_, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef,
                            clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    #print(pg_loss)
                    loss = pg_loss  -ent_coef * entropy_loss + v_loss * vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()

                if target_kl is not None:
                    if approx_kl > target_kl:
                        break
                #print('loss')
                #print(loss)

    y_pred, y_true = b_values.cpu().detach().numpy(), b_returns.cpu().detach().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    if rewards.mean().item() > best_rewards:
                        torch.save(agent.state_dict(), f"C:/Users/cherouss/Documents/12inter-20230429T172547Z-001/12inter/Models/SAMAPPO12/last/PY_LAST_Best_PPO_lstm_layers_{in_layer}_{run_name}_in_rewards")
                        print('****** new best rewards')
                        print(rewards.mean().item())
                        best_rewards = rewards.mean().item()
    if mean_waiting_time<best_system:
                        torch.save(agent.state_dict(), f"C:/Users/cherouss/Documents/12inter-20230429T172547Z-001/12inter/Models/SAMAPPO12/last/PY_LAST_Best_PPO_lstm_layers_{in_layer}_{run_name}_in_systemWaitingTime")
                        print('***** new best system waiting time')
                        print(mean_waiting_time)
                        best_system = mean_waiting_time
    k = 0
    for name, param in agent.conv_layers3.named_parameters():
                #print(name,param)
                if k > 1:
                    break
                    k+=1
                k = 0
    for name, param in agent.multihead_attention3.named_parameters():
                #print(name,param)
                if k > 1:
                    break
                k+=1




    if(mean_waiting_time > 3600):
                    global_step-=1
                    mean_waiting_time =0
                    agent_waiting_time =0
    else:
                    print(global_step)
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                    writer.add_scalar("losses/explained_variance", explained_var, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("rewards/reward",rewards.mean().item(), global_step)
                    writer.add_scalar("rewards/waiting_time",mean_waiting_time/900, global_step)
                    writer.add_scalar("rewards/agent_waiting_time",agent_waiting_time, global_step)
                    mean_waiting_time =0
                    agent_waiting_time =0
    envs.close()    
writer.close()


# In[10]:

"""
if 1 :
    ver = ''
   
    xml_name = f"xml/SAMAPPO_my_reward_high_last_0.6_test.xml"
    PATH = "Models/SAMAPPO12/Best_PPO_lstm_layers_120_1692195437_in_systemWaitingTime"
    PATH2 = "Models/SAMAPPO12/Best_PPO_lstm_layers_120_1692200289_in_systemWaitingTime"
    PATH3 = "Models/SAMAPPO12/Best_PPO_lstm_layers_120_1692207389_in_rewards"
    p = "Models/SAMAPPO12/best"
    last = "Models/SAMAPPO12/LAST_Best_PPO_lstm_layers_120_1692277577_in_systemWaitingTime"
    lastone = "Models/SAMAPPO12/LAST_Best_PPO_lstm_layers_120_1692284338_in_systemWaitingTime"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    num_steps = 3600
    in_layer = 120
    envs = SumoEnvironment(net_file=f'12inter{ver}/12inter.net.xml',
                  route_file=f'12inter{ver}/rou.rou.xml',sumo_seed = 42,
                  use_gui=True,
                 min_green = 4, max_green=60,delta_time = 4, output_file =xml_name,
                  num_seconds=3600, reward_fn="my_reward")
    in_layer = 120
    k = 0
 

    agent = Model(envs,in_layer).to(device)
    agent.load_state_dict(torch.load(lastone))
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
