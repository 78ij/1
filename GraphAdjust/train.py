import numpy as np
import os
from copy import deepcopy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

import time
import pickle
import random
import threading
import pickle
import math
import env_graph
import network
from time import strftime
import time
from replay_buffer import *

from sumtree import Memory
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt

with open('RL_train_data_100.pkl','rb') as f:
    train_data = pickle.load(f)
train_data_size = len(train_data)

env_list = []
for i in range(0, train_data_size):
    try:
        env_tmp = env_graph.ENV(train_data[i])
        if env_tmp.getboxcollision() != 0:
            env_list.append(env_tmp)
    except:
        continue
train_data_size = len(env_list)
print(train_data_size)
#env_instance = env.ENV(train_data)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 16
GAMMA = 0.95
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 50000
TARGET_UPDATE = 10
REWARD_PLOT_INTERVAL = 200
# prioritized experience replay
UPDATE_NN_EVERY = 1
UPDATE_MEM_EVERY = 20          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000     # how often to update the hyperparameters
BUFFER_SIZE = int(1e6)      # replay buffer size
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)

action_size_per_item = env_graph.ACTION_SIZE_PER_ITEM
item_max_size = env_graph.ITEM_MAX_SIZE

device = 'cuda:0'

policy_net = network.DQNetwork_simple(256, action_size_per_item * item_max_size).to(device)
target_net = network.DQNetwork_simple(256, action_size_per_item * item_max_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(),lr=0.0001)
#optimizer = optim.RMSprop(policy_net.parameters(),lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.9)
memory = ReplayBuffer(
            action_size_per_item * item_max_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, 12254151, True)

steps_done = 0

eps_list = []
eps_threshold = 1

def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def select_action(state, size):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    data,edge_index,edge_type = state
    length = [data.shape[0]]


    data = torch.tensor(data.astype("float32"),device=device)
    edge_index = torch.tensor(edge_index.astype("long"),device=device)
    edge_type = torch.tensor(edge_type.astype("long"),device=device)
    #print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_net.eval()
            out,_ = policy_net(data.reshape(1,-1,150),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3),length)
            policy_net.train()
            #print(out)
            #print(out.max(0))
            return out.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(size * action_size_per_item)]], device=device, dtype=torch.long)
 
def select_action_target(state):
    data,edge_index,edge_type = state
    #print(edge_index)
    length = [data.shape[0]]
    data = torch.tensor(data.astype("float32"),device=device)
    edge_index = torch.tensor(edge_index.astype("long"),device=device)
    edge_type = torch.tensor(edge_type.astype("long"),device=device)
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        #print(state.shape)
        out,_ = target_net(data.reshape(1,-1,150),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3),length)
       # print(out)
        #print(state.shape)
        #print(out)
        #print(out)
        #print(out.max(0))
        return out.max(1)[1].view(1, 1)

episode_durations = []
rewards_validation = []

def plot_durations():
    fig = plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    ax_cof = HostAxes(fig, [0.15, 0.1, 0.65, 0.8]) 

    ax_eps = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_rewards = ParasiteAxes(ax_cof, sharex=ax_cof)

    ax_cof.parasites.append(ax_eps)
    ax_cof.parasites.append(ax_rewards)

    ax_cof.axis['right'].set_visible(False)

    ax_eps.axis['right'].set_visible(True)
    ax_eps.axis['right'].major_ticklabels.set_visible(True)
    ax_eps.axis['right'].label.set_visible(True)

    ax_rewards.axis['right2'] = ax_rewards.new_fixed_axis(loc='right', offset=(60,0))

    fig.add_axes(ax_cof)
    
    ax_cof.plot(durations_t.numpy(), label='Duration')
    p1, = ax_eps.plot(np.array(eps_list),label='Epsilon')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 500:
        means = durations_t.unfold(0, 500, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(499), means))
        ax_cof.plot(means.numpy(),label='Mean Duration')

    x = np.array(list(range(0, len(rewards_validation)))) * REWARD_PLOT_INTERVAL
    #print(rewards_validation)
    #print(x)

    p2, = ax_rewards.plot(x,rewards_validation,label='Rewards')

    #p2, = ax_rewards.plot([41,42,43,44],label='Rewards')
    #set label for axis
    ax_cof.set_ylabel('Duration')
    ax_cof.set_xlabel('Episode')
    
    ax_eps.set_ylabel('Epsilon')
    ax_rewards.set_ylabel('Rewards')

    ax_cof.legend()

    ax_eps.axis["right"].label.set_color(p1.get_color())
    ax_rewards.axis["right2"].label.set_color(p2.get_color())

    ax_cof.set_ylim(0, 50)
    ax_eps.set_ylim(0, 1)
    ax_rewards.set_ylim(-100, 100)
    plt.pause(0.1)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
header = 'Time   Episode   LR     Loss'
start_time = time.time()
episode_now = 0

def merge_graph(state_batch):
    data_ret = []
    edge_index_ret = []
    edge_type_ret = []
    lengths = []
    offset = 0
    for s in state_batch:
        data_tmp,edge_index_tmp,edge_type_tmp = s
        edge_index_tmp = edge_index_tmp.T
        edge_type_tmp = edge_type_tmp.T
        lengths.append(data_tmp.shape[0])

        data_ret.append(data_tmp)
        edge_index_ret.append(edge_index_tmp + offset)
        offset += data_tmp.shape[0]
        edge_type_ret.append(edge_type_tmp)
    
    return np.concatenate(data_ret,axis=0).reshape(1,-1,150),np.concatenate(edge_index_ret,axis=0).reshape(1,-1,2),np.concatenate(edge_type_ret,axis=0).reshape(1,-1,3),lengths

def optimize_model():
    global episode_now
    episode_now += 1
    #if len(memory) < BATCH_SIZE:
    #    return
    #transitions = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones, weights, indices = memory.sample()

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # try:
    #     batch = Transition(*zip(*transitions))
    # except:
    #     print(transitions)
    #     return
    #print(batch.reward)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          next_states)), device=device, dtype=torch.bool)
    non_final_next_states = [s for s in next_states
                                                if s is not None]
    # state_batch = (batch.state)
    # action_batch = torch.cat(batch.action)
    # rewards = torch.cat(batch.reward)

    state_input, edge_index_input, edge_type_input,lengths = merge_graph(states)

    state_input = torch.tensor(state_input.astype("float32"),device=device)
    edge_index_input = torch.tensor(edge_index_input.astype("long"),device=device)
    edge_type_input = torch.tensor(edge_type_input.astype("long"),device=device)

    next_state_input, next_edge_index_input, next_edge_type_input,next_lengths = merge_graph(non_final_next_states)
    
    next_state_input = torch.tensor(next_state_input.astype("float32"),device=device)
    next_edge_index_input = torch.tensor(next_edge_index_input.astype("long"),device=device)
    next_edge_type_input = torch.tensor(next_edge_type_input.astype("long"),device=device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # idx = 0
    # state_action_values = torch.zeros(BATCH_SIZE, device=device)
    # for s in state_batch:
    #     data,edge_index,edge_type = s
    #     data_exist = np.zeros((20,1))
    #     data_exist[0:np.size(data,0),:] = np.ones((np.size(data,0),1))

    #     data_input = torch.tensor(data.astype("float32"),device=device)
    #     data_exist = torch.tensor(data_exist.astype("float32"),device=device)

    #     edge_index = torch.tensor(edge_index.astype("long"),device=device)
    #     edge_type = torch.tensor(edge_type.astype("long"),device=device)

    #     #print(policy_net(data_input.reshape(1,20,-1),data_exist.reshape(1,20,1),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3)).shape)
    #     #print(actions.shape)

    #     state_action_values[idx] = policy_net(data_input.reshape(1,20,-1),data_exist.reshape(1,20,1),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3))[0,action_batch[idx,0]]
    #     idx += 1

    state_action_values,loss_diff = policy_net(state_input.reshape(1,-1,150), edge_index_input.T.long().reshape(1,-1,2), edge_type_input.T.reshape(1,-1,3),lengths)
    state_action_values = state_action_values.gather(1, actions)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    #print(non_final_mask.shape)
    # tmp_next_state_value = torch.zeros(next_state_values[non_final_mask].shape[0],device=device)
    # idx = 0
    # for s in non_final_next_states:
    #     data,edge_index,edge_type = s
    #     data_exist = np.zeros((20,1))
    #     data_exist[0:np.size(data,0),:] = np.ones((np.size(data,0),1))

    #     data_input = torch.tensor(data.astype("float32"),device=device)
    #     data_exist = torch.tensor(data_exist.astype("float32"),device=device)

    #     edge_index = torch.tensor(edge_index.astype("long"),device=device)
    #     edge_type = torch.tensor(edge_type.astype("long"),device=device)
    #     tmp_next_state_value[idx] = target_net(data_input.reshape(1,20,-1),data_exist.reshape(1,20,1),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3)).max(1)[0].detach()
    #     idx += 1
    #print(tmp_next_state_value.shape)
    next_state_tmp,_ = target_net(next_state_input.reshape(1,-1,150), next_edge_index_input.T.long().reshape(1,-1,2), next_edge_type_input.T.reshape(1,-1,3),next_lengths)
    next_state_values[non_final_mask] = next_state_tmp.max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + rewards.flatten()
   # print(next_state_values.shape)
   # print(rewards.shape)
   # print(expected_state_action_values.shape)
   # print(state_action_values.shape)
    abs_errors = torch.abs(expected_state_action_values.flatten() - state_action_values.flatten())

    memory.update_priorities(abs_errors.detach().cpu().numpy(), indices)  
    #print(ISWeights.shape)
    #print(abs_errors)
    #ISWeights = torch.tensor(ISWeights,device=device,dtype=torch.float32).flatten()
    # Compute Huber loss
    #print(ISWeights)
    weights = torch.tensor(weights,device='cuda:0')
    #print(weights.shape)
   # print(weights)
    loss = F.smooth_l1_loss(state_action_values.flatten(), expected_state_action_values.flatten())
    # print('abs_errors ') 
    # print(abs_errors)
    # print('state_action_values ') 
    # print( state_action_values.flatten())
    # print('expected_state_action_values ')
    # print(expected_state_action_values.flatten())
    # print('next_state_values ')
    # print(next_state_values.flatten())
    # print('next_state_input ')
    # print(next_state_input)
    # print('rewards ')
    # print(rewards.flatten())
    # print('weights ')
    # print(weights)
   # loss = F.mse_loss( state_action_values.flatten(), expected_state_action_values.flatten())
    #print(state_action_values)
    #print(expected_state_action_values)
    #print(rewards)
    lr = optimizer.param_groups[0]['lr']
    print(
        f'''{strftime("%H:%M:%S", time.gmtime(time.time())):>9s} '''
        f'''{episode_now:>10}'''
        f'''{lr:>10.2E} '''
        f'''{loss.item():>10.2f}''')
     #   f'''{loss_diff.item():>10.2f}''')
    # Optimize the model
    optimizer.zero_grad()
    loss += loss_diff
    loss.backward()
    #torch.nn.utils.clip_grad_value_(policy_net.parameters(),1)
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()
    scheduler.step()


t_step_nn = 0
# Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
t_step_mem_par = 0
# Initialize time step (for updating every UPDATE_MEM_EVERY steps)
t_step_mem = 0
num_episodes = 50000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env_idx = i_episode % train_data_size
    env_list[env_idx].reset()


    state = env_list[env_idx].get_state()
    for t in range(50):
        # Select and perform an action
        action = select_action(state,env_list[env_idx].item_count_real)
        reward, done = env_list[env_idx].step(action.item())
        #reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = env_list[env_idx].get_state()
        else:
            next_state = None
        
        # Store the transition in memory
        #memory.store(Transition(state, action, next_state, reward))
        memory.add(state, action, reward, next_state, done)

        t_step_mem = (t_step_mem + 1) % UPDATE_MEM_EVERY
        t_step_mem_par = (t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if t_step_mem_par == 0:
            memory.update_parameters()

        # If enough samples are available in memory, get random subset and learn
        if memory.experience_count > EXPERIENCES_PER_SAMPLING:
            optimize_model()
        if t_step_mem == 0:
            memory.update_memory_sampling()

        #memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)

        if done:
            break
    episode_durations.append(t + 1)
    eps_list.append(eps_threshold)
    #print(eps_list)
    if i_episode % REWARD_PLOT_INTERVAL == 0:
        rewards = []
        for i in range(len(env_list)):
            env_list[i].reset()
            reward_tmp = []
            for t in range(50):
                # Select and perform an action
                state = env_list[i].get_state()
               # print(state[0][:,:7])
                action = select_action_target(state)
              #  print(action)
                reward, done = env_list[i].step(action.item())
            #    print(reward)

                reward_tmp.append(reward)
                if done:
                    break
            reward_final = 0

            for i in range(len(reward_tmp) - 1,-1,-1):
                reward_final = (reward_tmp[i] + GAMMA * reward_final)
            rewards.append(reward_final)
            #print(reward_tmp)
        print(rewards)
        rewards_validation.append(sum(rewards) / len(rewards))
        torch.save(target_net.state_dict(), './net.pth')
    plot_durations()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()
torch.save(target_net.state_dict(), './net.pth')