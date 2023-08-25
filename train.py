from agent.utils import *
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import os

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
num_agents = 6    
state_dim = 10 + num_agents
action_dim = 2
global_state_dim = (state_dim * num_agents) + num_agents 


    
class Model(nn.Module):
    def __init__(self, envs, in_layer = 64):
        super().__init__()
        
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
            layer_init(nn.Linear(in_layer, envs.action_space.n), std=0.01),
        )
        self.multihead_attention = nn.MultiheadAttention(state_dim, 2)
        #self.norm1 = nn.LayerNorm(state_dim)
        self.multihead_attention2 = nn.TransformerEncoderLayer(d_model=state_dim, nhead=2,dim_feedforward = 128,dropout = 0.1, batch_first  = True )
    def get_states(self, x):
        out = self.multihead_attention2(x)
        return out

    def get_value(self, x):
        
        return self.critic(x)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        #print(logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


def run():    
    run_name = f"{int(time.time())}"
    track = False
    name = f'SA_PPO_SUMO{run_name}'

    writer = SummaryWriter(f"SA6_{name}")
    
    

    

    seed = 42
    torch_deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    num_steps = 3600
    in_layer = 120
    envs = SumoEnvironment(net_file='data/6intersections/normal/6inter.net.xml',
                  route_file='data/6intersections/normal/rou.rou.xml',
                  use_gui=True,
                 min_green = 4, max_green=60,delta_time = 4,
                  num_seconds=3600, reward_fn="custom_reward")

    SAMAPPO = Model(envs,in_layer).to(device)
    learning_rate = 1e-4
    update_epochs = 20
    optimizer = optim.Adam(SAMAPPO.parameters(), lr=learning_rate, eps=1e-5)
    #assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    
    clip_coef=0.1

    # ALGO Logic: Storage setup
    num_envs = 1 
    
    update_freq = 900
    batch_size = update_freq * num_envs
    minibatch_size = 60 #//10
    
    obs = torch.zeros((update_freq, num_agents,state_dim)).to(device)
    obs2 = torch.zeros((update_freq, num_agents,state_dim)).to(device)
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

    for update in range(1, num_updates + 1):
        aa = True
        next_obs = torch.zeros((num_agents,state_dim)).to(device)
        next_done = torch.zeros((1)).to(device)
        envs.reset()
        print(envs.ts_ids)
        for tls in envs.ts_ids:
            actss[tls] = torch.zeros_like(actions2).to(device)
            probss[tls] = torch.zeros_like(logprobs2).to(device)
            n_actss[tls] = 0
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, 3600):
            global_step += 1 * num_envs
            x = SAMAPPO.get_states(next_obs)
            obs[up] = next_obs
            obs2[up] = x
            dones[up] = next_done
            acts = []
            probs = []
            with torch.no_grad():
                i=0 
                for tls in envs.ts_ids:
                    action, logprob, _,  = SAMAPPO.get_action(x[i])
                    acts.append(action)
                    probs.append(logprob)
                    i+=1
                    actss[tls][up] = action
                    probss[tls][up] = logprob
                    n_actss[tls] = action
            
                    
            act_dict = {}
            
                
            actions[up] = torch.Tensor(acts)
            logprobs[up] = torch.Tensor(probs)
            global_ob =  torch.cat((x.clone().view(-1),torch.tensor(acts)),0,)
            global_obs[up] =  global_ob
            values[up] = SAMAPPO.get_value(global_ob)
            
            
                

            next_obs, reward, done, info = envs.step(n_actss)
            
            next_obs = np.array(list(next_obs.values()))
            
            reward_ = np.array(list(reward.values())).sum()
            done = any(list(done.values()))
            mean_waiting_time +=  info["system_mean_waiting_time"]
            agent_waiting_time += info["agents_total_accumulated_waiting_time"]
            rewards[up] = torch.tensor(reward_).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
            up+=1
            
            if up == update_freq:
                infos2 = info
                print(infos2)
                break
        with torch.no_grad():
                next_value = SAMAPPO.get_value(global_ob)
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
                b_obs2= obs2.reshape((-1,) + (num_agents,state_dim))
                print(b_obs.shape)
                b_dones = dones.reshape(-1)
                
                
                b_logprobs = logprobs.reshape(-1,num_agents)
                b_logprobs2 = probss
                b_actions = actions.reshape((-1,num_agents))
                b_actions2 = actss
                b_advantages = advantages.reshape(-1)
                global_b_obs = global_obs.reshape((-1,global_state_dim))

                b_returns = returns.reshape(-1)

                b_values = values.reshape(-1)
                b_inds = np.arange(batch_size)
                clipfracs = [] 
        for epoch in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    logratio = torch.zeros_like(b_logprobs).to(device)
                    newvalue = SAMAPPO.get_value(global_b_obs[mb_inds])
                    newvalue = newvalue.view(-1)
                    loss = 0
                    i = 0
                    x = SAMAPPO.get_states(b_obs[mb_inds])
                    for tls in envs.ts_ids:
                        __, newlogprob, entropy = SAMAPPO.get_action(x[:,i], b_actions2[tls][mb_inds])
                        logratio[mb_inds,i] = (newlogprob - b_logprobs2[tls][mb_inds])
                        i+=1
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
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(SAMAPPO.parameters(), max_grad_norm)
                    optimizer.step()

                if target_kl is not None:
                    if approx_kl > target_kl:
                        break

        y_pred, y_true = b_values.cpu().detach().numpy(), b_returns.cpu().detach().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if rewards.mean().item() > best_rewards:
                    torch.save(SAMAPPO.state_dict(), f"SA6_{in_layer}_{run_name}_in_rewards")
                    print('\n******************** new best rewards ****************************')
                    print(rewards.mean().item())
                    best_rewards = rewards.mean().item()        



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
run()
