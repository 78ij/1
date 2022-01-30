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
import env
import network
from time import strftime
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt

with open('RL_train_data.pkl','rb') as f:
    train_data = pickle.load(f)

env_instance = env.ENV(train_data)



BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 10
REWARD_PLOT_INTERVAL = 10
SEQ_LEN = 50
HIDDEN_SIZE = 2048

action_size_per_item = env.ACTION_SIZE_PER_ITEM
item_max_size = env.ITEM_MAX_SIZE

device = 'cuda:0'

policy_net = network.DQNetwork_simple(150 * item_max_size, HIDDEN_SIZE, action_size_per_item * item_max_size).to(device)
target_net = network.DQNetwork_simple(150 * item_max_size, HIDDEN_SIZE, action_size_per_item * item_max_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = network.ReplayMemory(10000)


steps_done = 0

eps_list = []
eps_threshold = 1
     
def select_action_target(state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        out = policy_net(state)
        #print(out)
        #print(out.max(0))
        return out.max(0)[1].view(1, 1)

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
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
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
    ax_rewards.set_ylim(-1000, 1000)
    plt.pause(0.1)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
header = 'Time   Episode        Loss'
start_time = time.time()
episode_now = 0
def optimize_model():
    global episode_now
    episode_now += 1
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = network.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.view(-1,150 * item_max_size)).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.view(-1,150 * item_max_size)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    print(
        f'''{strftime("%H:%M:%S", time.gmtime(time.time())):>9s} '''
        f'''{episode_now:>10.2f}'''
        f'''{loss.item():>10.2f}''')
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

z_state = torch.zeros([1,BATCH_SIZE,int(HIDDEN_SIZE / 2)],device=device,dtype=torch.float32 )
h_state = (z_state,z_state)

init_state = (z_state,z_state)
num_episodes = 5000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env_instance.reset()
    ended = np.array([False] * BATCH_SIZE)
    # data for the time steps
    state_list = []
    finish_tag_list = []

    masks = np.zeros([BATCH_SIZE, SEQ_LEN])
    action_mask = np.zeros([BATCH_SIZE, SEQ_LEN, action_size_per_item * item_max_size])
    Q_target = np.zeros([BATCH_SIZE, SEQ_LEN])
    traj_length = np.zeros(BATCH_SIZE).astype(np.int32)
    h_state = init_state
    rewards_list = []

    for t in range(SEQ_LEN):
        # Select and perform an action
        state = env_instance.get_state()
        state_list.append(state) 
        #finish_tag_list.append(tag)

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        #print(eps_threshold)

        out, h_state = target_net(torch.tensor(state, device=device,dtype=torch.float32), h_state)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(out)
                #print(out.max(0))
                action =  out.flatten().max(0)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(item_max_size * action_size_per_item)]], device=device, dtype=torch.long)

        reward, done = env_instance.step(action.item())
        reward = torch.tensor([reward], device=device)
        
        for i,e in enumerate(ended):
                if e:
                    reward[i] = 0

        rewards_list.append(reward)

        traj_length += (t+1) * np.logical_and(ended == 0, (done == 1))

        ended = np.logical_or(ended, (done == 1))

        if ended.all():
            break

        if done:
            break

    out, h_state = target_net(torch.tensor(state, device=device,dtype=torch.float32), h_state)
    last_value = out.flatten().max(0)[0] # bs
    discount_reward = np.zeros(BATCH_SIZE, np.float32)

    #print(last_value)
    #print(rewards_list)
    #print(last_value)
    for i in range(BATCH_SIZE):
        if not ended[i]:
            discount_reward[i] = last_value

    length = len(rewards_list)

    for x in range(length-1, -1, -1):
        discount_reward = rewards_list[x].detach().cpu() + GAMMA * discount_reward

        Q_target[:,x] = discount_reward
    #print(Q_target)

    traj_length += SEQ_LEN * (ended == 0)
    for i,l in enumerate(traj_length):
        masks[i,:l] = 1

    for i in range(SEQ_LEN - len(state_list)):
        state_list.append(np.zeros_like(state_list[0]))
        #finish_tag_list.append(np.zeros_like(finish_tag_list[0]))

    inputs_ = torch.tensor(state_list, device=device,dtype=torch.float32).reshape(-1,1,150 * item_max_size) # seqlen * 1  * featurelen
    #print(t)
    episode_durations.append(t + 1)
    eps_list.append(eps_threshold)
   # print(inputs_.shape)
    out_train, out_state = policy_net(inputs_, init_state)
    Q_target = torch.tensor(Q_target, device=device,dtype=torch.float32)
    print(Q_target)
    print(out_train)
    out_train = out_train.max(2)[0]
    print(out_train)
    loss = F.smooth_l1_loss(out_train.flatten(), Q_target.flatten())
    print(
        f'''{strftime("%H:%M:%S", time.gmtime(time.time())):>9s} '''
        f'''{episode_now:>10.2f}'''
        f'''{loss.item():>10.2f}''')
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    #print(eps_list)

    
    if i_episode % REWARD_PLOT_INTERVAL == 0:
        env_instance.reset()
        reward_tmp = []
        h_state = (z_state,z_state)
        for t in range(50):
            # Select and perform an action
            state = torch.tensor(env_instance.get_state(),device=device,dtype=torch.float32)
            #action = select_action_target(state)
            out, h_state = target_net(torch.tensor(state, device=device,dtype=torch.float32), h_state)
            #print(out.shape)
            action =  out.flatten().max(0)[1].view(1, 1)

            reward, done = env_instance.step(action.item())
            reward_tmp.append(reward)
            if done:
                break
        reward_final = 0

        for i in range(len(reward_tmp) - 1,-1,-1):
            reward_final = (reward_tmp[i] + GAMMA * reward_final)
        rewards_validation.append(reward_final)
    plot_durations()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()
torch.save(target_net.state_dict(), './net.pth')