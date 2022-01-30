import numpy as np
import os
from copy import deepcopy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


import time
import pickle
import copy
import random
import threading
import pickle
import math
import env_subgraph
import network_subgraph
from time import strftime
import time
from replay_buffer_openai import PrioritizedReplayBuffer 
from linear_schedule import LinearSchedule

from sumtree import Memory
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import imageio
import matplotlib.pyplot as plt
from MCTS import *
import network_discriminator
import traceback
from multiprocessing.dummy import Pool, Process, Lock
print('xxx')
with open('RL_train_data_ldroom_complete_subbox_2.pkl','rb') as f:
    train_data = pickle.load(f)
train_data_size = len(train_data)

tmp_env_list = []
env_list = []
action_list =  []        
for i in range(train_data_size):
    print(i)
    try:
        env_tmp = env_subgraph.ENV(train_data[i])
        coll, _ = env_tmp.getboxcollision()
        if coll != 0:
            tmp_env_list.append(env_tmp)
            if len(tmp_env_list) == 100: break
    except:
        continue
actlst = os.listdir('./action_heru_ldroom/')
actlst = [int(x[:-7]) for x in actlst]
actlst.sort()
idx = 0
for act in range(76,77):
    with open('./action_heru_ldroom/' + str(act) + '.action','rb') as f:
        action_list.append(pickle.load(f))
    env_list.append(tmp_env_list[act])

    print(act)

   # X = env_list[-1].visualize2D()
    
   # plt.imshow(X)
   # plt.show()

#with open('saved_train_data.pkl','wb') as f:
 #   pickle.dump(env_list,f)
print(action_list)
val_env_list = env_list#[500:]
env_list = env_list#[:500]
train_data_size = len(env_list)
print(train_data_size)
#env_instance = env.ENV(train_data)
Transition = namedtuple('Transition',
                        ('state', 'true_action','reward','next_state','done'))


LR = 0.0001
BATCH_SIZE = 16
GAMMA = 0.95
EPS_START = 0.7
EPS_END = 0.05
EPS_DECAY = 8000000
TARGET_UPDATE = 100
REWARD_PLOT_INTERVAL = 2000
# prioritized experience replay
UPDATE_NN_EVERY = 1
UPDATE_MEM_EVERY = 20   # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 5000     # how often to update the hyperparameters
BUFFER_SIZE = int(1e7)      # replay buffer size
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)

action_size_per_item = env_subgraph.ACTION_SIZE_PER_ITEM
item_max_size = env_subgraph.ITEM_MAX_SIZE

device = 'cuda:0'

policy_net = network_subgraph.DQNetwork_simple(512).to(device)
target_net = network_subgraph.DQNetwork_simple(512).to(device)
policy_net.load_state_dict(torch.load('./net_imitationa_all.pth'))

#D_net = network_discriminator.Network_Dis(512).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(),lr=LR, weight_decay=0.00001)
#optimizer_D = optim.Adam(D_net.parameters(),lr=LR, weight_decay=0.00001)

#optimizer = optim.RMSprop(policy_net.parameters(),lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000, gamma=0.95)
# memory = ReplayBuffer(
#             action_size_per_item * item_max_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, 12254151, True)
memory = PrioritizedReplayBuffer(300000, alpha=0.4)
beta_schedule = LinearSchedule(100000,
                                initial_p=0.2,
                                final_p=0.8)
steps_done = 0

eps_list = []
eps_threshold = 1

def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
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
    #eps_threshold = EPS_START
    steps_done += 1
    print('eps ' + str(eps_threshold))
    all_data,all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wall_countour = state
    #print(edge_index)
    length = [len(subbox_lengths)]

    all_data = torch.tensor(all_data.astype("float32"),device=device)
    all_edge = torch.tensor(all_edge.astype("long"),device=device)
    all_type = torch.tensor(all_type.astype("float32"),device=device)
    data = torch.tensor(data.astype("float32"),device=device)
    edge_index = torch.tensor(edge_index.astype("long"),device=device)
    edge_type = torch.tensor(edge_type.astype("float32"),device=device)
    wall_countour = torch.tensor(wall_countour.astype("float32"),device=device)
    #print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            out,_ = policy_net(all_data.reshape(1,-1,7),all_edge.T.reshape(1,-1,2), all_type.T.reshape(1,-1,3),data.reshape(1,-1,11), edge_index.long().T.reshape(1,-1,2), edge_type.T.reshape(1,-1,3),length,subbox_lengths,wall_countour)
            #print(out)
            #print(out.max(0))
            return (out).max(1)[1].view(1, 1)
    else:
        feasible = (wall_countour < 0.000001)
        feasible = feasible.flatten()
        while True:
            action_proposed = random.randrange(size * action_size_per_item)
            if int(action_proposed/4) < (size - 1) and feasible[action_proposed] == 0:
                return torch.tensor([[action_proposed]], device=device, dtype=torch.long)
            elif int(action_proposed/4) >= (size - 1):
                return torch.tensor([[action_proposed]], device=device, dtype=torch.long)
def select_action_target(state):
    
    all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wall_countour = state    #print(edge_index)
    length = [len(subbox_lengths)]
    all_edge = torch.tensor(all_edge.astype("long"),device=device)
    all_type = torch.tensor(all_type.astype("float32"),device=device)
    data = torch.tensor(data.astype("float32"),device=device)
    edge_index = torch.tensor(edge_index.astype("long"),device=device)
    edge_type = torch.tensor(edge_type.astype("float32"),device=device)
    wall_countour = torch.tensor(wall_countour.astype("float32"),device=device)
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        #print(state.shape)
        out,_ = policy_net(all_edge.T.reshape(1,-1,2), all_type.T.reshape(1,-1,3),data.reshape(1,-1,11), edge_index.long().T.reshape(1,-1,2), edge_type.T.reshape(1,-1,3),length,subbox_lengths,wall_countour)
        print(out)
        #print(state.shape)
        #print(out)
    #    print(out)
        #print(out.max(0))
        return out.max(1)[1].view(1, 1)
def select_action_target_mcts(state,env):
    
    data,edge_index,edge_type,subbox_lengths = state
    #print(edge_index)
    length = [len(subbox_lengths)]
    data = torch.tensor(data.astype("float32"),device=device)
    edge_index = torch.tensor(edge_index.astype("long"),device=device)
    edge_type = torch.tensor(edge_type.astype("float32"),device=device)
    
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        #print(state.shape)
        out,_ = target_net(state_input.reshape(1,-1,11), edge_index_input.long().reshape(1,-1,2), edge_type_input.reshape(1,-1,3),lengths,subbox_lengths,wall_countours.reshape(-1,1024*3))
        #print(out)
        #print(state.shape)
        #print(out)
        #print(out)
        #print(out.max(0))
        
        mct = MCT()
        mct.root.state = env.get_state_for_mcts()
        max_value = out.max(1)[0].view(1, 1)
        mct.setactionsize(action_size_per_item * item_max_size)
        mct.root.value = max_value
        for i in range(20):
            env.set_state(mct.root.state)
            _id = mct.selection()
            #simulation stage
            for j in range(20):
                
                _,data,edge_index,edge_type,subbox_lengths = mct.getstate(_id)
                length = [len(subbox_lengths)]
                data = torch.tensor(data.astype("float32"),device=device)
                edge_index = torch.tensor(edge_index.astype("long"),device=device)
                edge_type = torch.tensor(edge_type.astype("float32"),device=device)
                out,_ = target_net(data.reshape(1,-1,10),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3),length,subbox_lengths)
                max_action = out.max(1)[1].view(1, 1)
                max_value = out.max(1)[0].view(1, 1)

                reward, done = env.step(max_action.item())
                #print(reward)
                succ, _id = mct.expansion(_id, max_action, reward, env.get_state_for_mcts(),done)
                if succ:
                    mct.nodedict[_id].value = max_value
                    mct.backpropagation(_id)
                if done: break
        env.set_state(mct.root.state)
        mct.root.printnode()    
        return mct.root.best


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
    if len(durations_t) >= 1000:
        means = durations_t.unfold(0, 999, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(999), means))
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


header = 'Time   Episode   LR     Loss'
start_time = time.time()


def merge_graph(state_batch):
    all_data = []
    all_edge_idx = []
    all_edge_type = []
    data_ret = []
    edge_index_ret = []
    edge_type_ret = []
    lengths = []
    subbox_lengths = []
    wall_countours = []
    offset = 0
    offset_upper = 0
    for s in state_batch:
        all_data_tmp, all_edge_idx_tmp, all_edge_type_tmp, data_tmp,edge_index_tmp,edge_type_tmp,subbox_lengths_tmp,wall_countour_tmp = s

        all_edge_idx_tmp = all_edge_idx_tmp.T
        all_edge_type_tmp = all_edge_type_tmp.T
     #   print(all_edge_type_tmp)
       # print(all_edge_idx_tmp.shape)
      #  print(all_edge_type_tmp.shape)
        edge_index_tmp = edge_index_tmp.T
        edge_type_tmp = edge_type_tmp.T
        lengths.append(len(subbox_lengths_tmp))
        subbox_lengths += subbox_lengths_tmp
        
        all_edge_idx.append(all_edge_idx_tmp + offset_upper)
        all_edge_type.append(all_edge_type_tmp)
        offset_upper += lengths[-1]
        data_ret.append(data_tmp)
        all_data.append(all_data_tmp)
        edge_index_ret.append(edge_index_tmp + offset)
        offset += data_tmp.shape[0]
        edge_type_ret.append(edge_type_tmp)
        wall_countours.append(wall_countour_tmp)
       # print(data_ret)
       # print(edge_index_ret)
       # print(edge_type_ret)
    # try:
    #     np.concatenate(data_ret,axis=0).reshape(1,-1,11),np.concatenate(edge_index_ret,axis=0).reshape(1,-1,2),np.concatenate(edge_type_ret,axis=0).reshape(1,-1,3),lengths,subbox_lengths
    # except:
    #     print(data_ret)
    #     print(edge_index_ret)
    #     print(edge_type_ret)
    #     exit(1)

    return np.concatenate(all_data,axis=0).reshape(1,-1,7),np.concatenate(all_edge_idx,axis=0),np.concatenate(all_edge_type,axis=0), np.concatenate(data_ret,axis=0).reshape(1,-1,11),np.concatenate(edge_index_ret,axis=0).reshape(1,-1,2),np.concatenate(edge_type_ret,axis=0).reshape(1,-1,3),lengths,subbox_lengths,np.concatenate(wall_countours,axis=0)

def optimize_model(x=False):
    global episode_now
    episode_now += 1
    #if len(memory) < BATCH_SIZE:
    #    return
    #transitions = memory.sample(BATCH_SIZE)
    states, actions,rewards,next_states,dones,is_im,weights,idxs = memory.sample(BATCH_SIZE, beta=beta_schedule.value(episode_now))
    #print(idx)
    rewards = torch.tensor(rewards.astype("float32"),device='cuda:0')
    weights = torch.tensor(weights.astype("float32"),device='cuda:0')
    print('rewards' + str(rewards))
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
    # rewards = torch.cat(batch.reward)
    non_final_mask = ~torch.tensor(dones, device=device, dtype=torch.bool)
    print('dones ' + str(dones))
   
    print('non final ' + str(non_final_mask))
    final_mask = (~(non_final_mask)).float()
    non_final_mask = torch.nonzero(non_final_mask).flatten()
    print('non final ' + str(non_final_mask))
 #   print('final ' + str(final_mask))

    non_final_next_states = []
  #  print(dones)
    for i in range(len(dones)):
        if not dones[i]:
            non_final_next_states.append(next_states[i])
    all_data, all_edge, all_type , state_input, edge_index_input, edge_type_input,lengths,subbox_lengths,wall_countours= merge_graph(states)

    all_data = torch.tensor(all_data.astype("float32"),device=device)
    all_edge = torch.tensor(all_edge.astype("long"),device=device)
    all_type = torch.tensor(all_type.astype("float32"),device=device)
    state_input = torch.tensor(state_input.astype("float32"),device=device)
    edge_index_input = torch.tensor(edge_index_input.astype("long"),device=device)
    edge_type_input = torch.tensor(edge_type_input.astype("float32"),device=device)
    wall_countours = torch.tensor(wall_countours.astype("float32"),device=device)
    is_im = torch.tensor(is_im.astype("float32"),device=device)
    print(is_im)
   # print(non_final_next_states)
    next_state_values = torch.zeros(BATCH_SIZE, device=device).detach()

    if(len(non_final_next_states) != 0):
        all_data_next, all_edge_next, all_type_next, next_state_input, next_edge_index_input, next_edge_type_input,next_lengths,next_subbox_lengths,next_wall_countours = merge_graph(non_final_next_states)
        
        all_data_next = torch.tensor(all_data_next.astype("float32"),device=device)
        all_edge_next = torch.tensor(all_edge_next.astype("long"),device=device)
        all_type_next = torch.tensor(all_type_next.astype("float32"),device=device)
        next_state_input = torch.tensor(next_state_input.astype("float32"),device=device)
        next_edge_index_input = torch.tensor(next_edge_index_input.astype("long"),device=device)
        next_edge_type_input = torch.tensor(next_edge_type_input.astype("float32"),device=device)
        next_wall_countours = torch.tensor(next_wall_countours.astype("float32"),device=device)
        next_state_tmp,_ = target_net(all_data_next.reshape(1,-1,7),all_edge_next.reshape(1,-1,2), all_type_next.reshape(1,-1,3), next_state_input.reshape(1,-1,11), next_edge_index_input.long().reshape(1,-1,2), next_edge_type_input.reshape(1,-1,3),next_lengths,next_subbox_lengths,next_wall_countours)
        next_state_tmp_value,_ = policy_net(all_data_next.reshape(1,-1,7),all_edge_next.reshape(1,-1,2), all_type_next.reshape(1,-1,3),next_state_input.reshape(1,-1,11), next_edge_index_input.long().reshape(1,-1,2), next_edge_type_input.reshape(1,-1,3),next_lengths,next_subbox_lengths,next_wall_countours)
        next_state_tmp = next_state_tmp
        next_state_tmp_value = next_state_tmp_value
        max_actions = (next_state_tmp_value).max(1)[1]
        next_state_values[non_final_mask] = next_state_tmp.gather(1,max_actions.unsqueeze(1)).flatten()
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
    actions = torch.tensor(actions,device='cuda:0')
    state_action_values,loss_diff = policy_net(all_data.reshape(1,-1,7),all_edge.reshape(1,-1,2), all_type.reshape(1,-1,3), state_input.reshape(1,-1,11), edge_index_input.long().reshape(1,-1,2), edge_type_input.reshape(1,-1,3),lengths,subbox_lengths,wall_countours)
   # print(state_action_values)
   # print(state_action_values.shape)
 #   print(actions)
  #  print(state_action_values)
    state_action_values_true = state_action_values.gather(1, actions)
    print('stateactionvalues_true' + str(state_action_values_true))

    
   # print(next_state_tmp.shape)
   # print(actions.shape)
   
   
    expected_state_action_values = (next_state_values * GAMMA) + rewards.flatten()
    loss1 = F.smooth_l1_loss(weights * state_action_values_true.flatten(),weights *  expected_state_action_values.flatten())
    print('stateactionvalues_expected' + str(expected_state_action_values))
    print('weights' + str(weights))

   # done_net_out = D_net(all_edge.reshape(1,-1,2), all_type.reshape(1,-1,3), state_input.reshape(1,-1,11), edge_index_input.long().reshape(1,-1,2), edge_type_input.reshape(1,-1,3),lengths,subbox_lengths,wall_countours)
  #  print('done_net_out: ' + str(done_net_out))
 #   loss_D = F.binary_cross_entropy(done_net_out,final_mask)
  
  #  optimizer_D.zero_grad()
    #loss += loss_diff
  #  loss_D.backward()
  #  print('Dloss: ' + str(loss_D))
    #torch.nn.utils.clip_grad_value_(policy_net.parameters(),1)
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
  #  optimizer_D.step()

  #  print(rewards)
    #print(state_action_values.shape)
    #print(state_action_values)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
   
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

    # Compute the expected Q values
   # print(next_state_values.shape)
   # print(rewards.shape)
   # print(expected_state_action_values.shape)
   # print(state_action_values.shape)

    #print(ISWeights.shape)
    #print(abs_errors)
    #ISWeights = torch.tensor(ISWeights,device=device,dtype=torch.float32).flatten()
    # Compute Huber loss
    #print(ISWeights)
    #print(weights.shape)
   # print(weights)

    #loss = F.smooth_l1_loss(state_action_values.flatten(), expected_state_action_values.flatten())
    

    margin_data = (torch.ones_like(state_action_values,device='cuda:0') * 5).detach()
    margin_data[torch.tensor(list(range(state_action_values.shape[0])),device='cuda:0'),actions.flatten()] = 0
    state_action_values_2 = state_action_values + margin_data
    #print(state_action_values_2)

    #print(margin_data)
    #print(state_action_values_2)
 #   print(state_action_values_true)
    print('add margined ' + str(state_action_values_2.max(1)[0]))
    loss = F.smooth_l1_loss(is_im * weights * state_action_values_2.max(1)[0].flatten(), is_im * weights * state_action_values_true.flatten())
    loss = loss + loss1
    #if(loss > 2):
    #    print(state_action_values)
    abs_errors = torch.abs(state_action_values_true.flatten() - expected_state_action_values.flatten())
    abs_errors +=  torch.abs(is_im * state_action_values_2.max(1)[0].flatten()- is_im  * state_action_values_true.flatten())
   # abs_errors2 = torch.abs(is_im  * state_action_values_2.max(1)[0].flatten() -  is_im  * state_action_values_true.flatten())
   # print(abs_errors.shape)
   # print(abs_errors2.shape)
    #abs_errors += abs_errors2
  #  print(idxs)
    print('abserror ' + str(abs_errors.detach().cpu().numpy().tolist()))
    memory.update_priorities(idxs, abs_errors.detach().cpu().numpy().tolist())  
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
    # print(
    #     f'''{strftime("%H:%M:%S", time.gmtime(time.time())):>9s} '''
    #     f'''{episode_now:>10}'''
    #     f'''{lr:>10.2E} '''
    #     f'''{loss.item():>10.2f}''')
       # f'''{loss_diff.item():>10.2f}''')
    # Optimize the model
   # if not x:
    print('optimizing')
    optimizer.zero_grad()
    #loss += loss_diff
    loss.backward()
    #torch.nn.utils.clip_grad_value_(policy_net.parameters(),1)
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()
    print(lr,loss)
    scheduler.step()
   #else:
     #   print(all_edge, all_type , state_input, edge_index_input, edge_type_input,lengths,subbox_lengths,wall_countours)


t_step_nn = 0
# Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
t_step_mem_par = 0
# Initialize time step (for updating every UPDATE_MEM_EVERY steps)
t_step_mem = 0
for i_episode in  range(len(env_list)):
    # Initialize the environment and state

   # env_idx = random.randint(0,len(env_list) - 1)
    env_idx = i_episode
    env_list[env_idx].reset()


    print(action_list[i_episode])
    for t in range(len(action_list[i_episode])):
        # Select and perform an action
        state = env_list[env_idx].get_state()
        action = action_list[i_episode][t]
        reward, done = env_list[env_idx].step(action)
       # if t == len(action_list[i_episode]) - 1: done = True
        print(reward,done)
        #reward = torch.tensor([reward], device=device)
        
        # Store the transition in memory
        #memory.store(Transition(state, action, next_state, reward))
        nstate = env_list[env_idx].get_state()

        memory.add(state, action, reward, nstate, done,True)
    #input()
idx_step = 0
print('Memory Length: ' + str(len(memory)))
i_episode = 0
memory.start_real_sample()
lock = Lock()
def train_iteration(i_episode):
    global memory
    global train_data_size
    global env_list
    global idx_step
    if idx_step > 5:
        env_idx = i_episode % train_data_size
        env_list[env_idx].reset()
        for t in range(30):
            
            #print(t)y
            # Select and perform an action
            state = env_list[env_idx].get_state()
            action = select_action(state,env_list[env_idx].item_count_real+1)
            reward, done = env_list[env_idx].step(action.item())
            #print(reward)
            #reward = torch.tensor([reward], device=device)

            # Observe new state
            #if not done:
            next_state = env_list[env_idx].get_state()
          #  else:
           #     next_state = None
            lock.acquire()
            if idx_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if idx_step % 500 == 0:
                torch.save(policy_net.state_dict(), './net_imitationab_all.pth')
              #  torch.save(D_net.state_dict(), './net_discriminator.pth')

            # Store the transition in memory
            #memory.store(Transition(state, action, next_state, reward))
            memory.add(state, action.item(), reward, next_state, done, False)
            print('nextidx: ' + str(memory._next_idx) + ' real_sample:' + str(memory._is_real_sample))

            idx_step = idx_step + 1
            optimize_model()
            lock.release()
            if done: break
    else:
        lock.acquire()
        idx_step = idx_step + 1
        optimize_model()
        if idx_step % 500 == 0:
            torch.save(policy_net.state_dict(), './net_imitation_all.pth')
           # torch.save(D_net.state_dict(), './net_discriminator.pth')
            print('aaaaaaaaaaaaaaaaaaaaaaa')
        if idx_step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        lock.release()

    
for i in range(1000000):
    train_iteration(i)

# with Pool(processes=5) as pool:
#     pool.map(train_iteration, range(1000000))