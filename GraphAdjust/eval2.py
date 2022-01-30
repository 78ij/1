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
import env_subgraph
import network_subgraph
import network_discriminator

from time import strftime
import time
from replay_buffer import *
import imageio
from sumtree import Memory
from MCTS import *
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt
from scene_graph_structurenet import BasicSceneGraph

print('xxx')
with open('baseline_sync_living.pkl','rb') as f:
    train_data = pickle.load(f)
train_data_size = len(train_data)

tmp_env_list = []
env_list = []
action_list =  []        
for i in range(1, train_data_size):
    print(i)
    try:
        env_tmp = env_subgraph.ENV(train_data[i])
        coll, _ = env_tmp.getboxcollision()
        if coll != 0:
            tmp_env_list.append(env_tmp)
            if len(tmp_env_list) == 100: break
    except:
        continue
actlst = os.listdir('./saved_operations/')
actlst = [int(x[:-7]) for x in actlst]
actlst.sort()
idx = 0
env_list = tmp_env_list
#env_list = [tmp_env_list[75]]

# for act in (actlst):
#     with open('./saved_operations/' + str(act) + '.action','rb') as f:
#         action_list.append(pickle.load(f))
#     env_list.append(tmp_env_list[act])

    #print(act)
   # X = env_list[-1].visualize2D()
    
   # plt.imshow(X)
   # plt.show()



action_size_per_item = env_subgraph.ACTION_SIZE_PER_ITEM
item_max_size = env_subgraph.ITEM_MAX_SIZE

device = 'cuda:0'


policy_net = network_subgraph.DQNetwork_simple(512).to(device)
policy_net.load_state_dict(torch.load('./net_imitation.pth'))
#D_net = network_discriminator.Network_Dis(512).to(device)
#D_net.load_state_dict(torch.load('./net_d.pth'))

policy_net.eval()
num_episodes = 50000

GAMMA=0.95

def select_action_target(state):
    
    all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wallcountour = state
    #print(edge_index)
    length = [len(subbox_lengths)]
    all_edge = torch.tensor(all_edge.astype("long"),device=device)
    all_type = torch.tensor(all_type.astype("float32"),device=device)
    data = torch.tensor(data.astype("float32"),device=device)
    edge_index = torch.tensor(edge_index.astype("long"),device=device)
    edge_type = torch.tensor(edge_type.astype("float32"),device=device)
    wallcountour = torch.tensor(wallcountour.astype("float32"),device=device)
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        #print(state.shape)
        #print()
        out,_ = policy_net(all_edge.T.reshape(1,-1,2), all_type.T.reshape(1,-1,3),data.reshape(1,-1,11), edge_index.long().T.reshape(1,-1,2), edge_type.T.reshape(1,-1,3),length,subbox_lengths,wallcountour)
       
        #print(out.max(1)[1])
        #print(state.shape)
        #print(out)
       # print(out)
        #print(out.max(0))
        return out.max(1)[1].view(1, 1)

#[-38.72673756423049, 69.4731259866283, 500.0, -49.98625305822821, -5.4491737503424815, 51.651838225165925, -57.13651272914748, -406.48636768627233, 205.92000000000004, -36.29877675157941]
# Initialize the environment and state
rewards = []

mcts_config = {
    "puct_coefficient":5.0,
    "num_simulations": 20,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": True,
    "add_dirichlet_noise": False,
}

mcts = MCTS(policy_net,mcts_config)


for i in range(len(env_list)):
    total_state_value = []
    print('-------------------------------------------------')
    frames = []
    env_list[i].reset()
    #env_list[i].visualize()
    reward_tmp = []
    X = env_list[i].visualize2D()
    #plt.imshow(X)
   # plt.show()
    for t in range(30):
        X = env_list[i].visualize2D()
      #  plt.imshow(X)
    #    plt.show()
        frames.append(X)
        print(env_list[i].getboxcollision()[0])
        print(env_list[i].getwallcollision() * -5)
        print(env_list[i].get_free_space())
        total_state_value.append(env_list[i].getboxcollision()[0] * -5 + env_list[i].getwallcollision() * -5 + env_list[i].get_free_space())
       # plt.imshow(X)
      #  plt.show()
        # Select and perform an action
        state = env_list[i].get_state()
        # print(state[0][:,:7])
       # mcts_policy, action, tree_node = mcts.compute_action(tree_node)
        action = select_action_target(state)
      #  print(action)
      #  print(mcts_policy)
    #     #  print(action)
     #   env_list[i].set_state(state)
        reward, done = env_list[i].step(action.item())
      #  print(reward)

        all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wallcountour = env_list[i].get_state()
    #print(edge_index)
        length = [len(subbox_lengths)]
        all_edge = torch.tensor(all_edge.astype("long"),device=device)
        all_type = torch.tensor(all_type.astype("float32"),device=device)
        data = torch.tensor(data.astype("float32"),device=device)
        edge_index = torch.tensor(edge_index.astype("long"),device=device)
        edge_type = torch.tensor(edge_type.astype("float32"),device=device)

        wallcountour = torch.tensor(wallcountour.astype("float32"),device=device)
       # out2 = D_net(all_edge.T.reshape(1,-1,2), all_type.T.reshape(1,-1,3),data.reshape(1,-1,11), edge_index.long().T.reshape(1,-1,2), edge_type.T.reshape(1,-1,3),length,subbox_lengths,wallcountour)
      #  print(out)
        #print(out2)
        reward_tmp.append(reward)
       # env_list[i].set_state(tree_node.state)


    #     #env_list[i].visualize()
        if done:
            break
    #s    exit(1)
    X = env_list[i].visualize2D()
   # plt.imshow(X)
   # plt.show()
    total_state_value.append(env_list[i].getboxcollision()[0] * -5 + env_list[i].getwallcollision() * -5 + env_list[i].get_free_space())
    frames.append(X)
    #print(frames)
    total_state_value = np.array(total_state_value)
    idx_best = np.argmax(total_state_value)
    print(total_state_value)
   # frames = frames[:idx_best+1]
    imageio.mimsave('./tmp2/' + str(i) + '.gif', frames, 'GIF', fps=5, duration=0.075)
    #env_list[i].visualize()
    reward_final = 0
    for x in range(len(reward_tmp) - 1,-1,-1):
        reward_final = (reward_tmp[x] + GAMMA * reward_final)
    rewards.append(reward_final)
    print(reward_tmp)
    print(reward_final)
print(rewards)
