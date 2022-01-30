import numpy as np
import os
from copy import deepcopy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from multiprocessing.dummy import Pool, Process, Lock
# if __name__ == '__main__':
#     set_start_method('spawn')

import sys
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

print('xxx')
with open('RL_train_data_bedroom_complete_subbox_2.pkl','rb') as f:
    train_data = pickle.load(f)
train_data_size = len(train_data)

env_list = []
action_list =  []        
for i in range(1, train_data_size):
    print(i)
    try:
        env_tmp = env_subgraph.ENV(train_data[i])
        coll, _ = env_tmp.getboxcollision()
        if coll != 0:
            env_list.append(env_tmp)
            if len(env_list) == 30: break
    except:
        continue
val_env_list = env_list#[500:]
train_data_size = len(env_list)


def select_action(env):

    all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wall_countour = env.get_state()
    #print(edge_index)
    length = [len(subbox_lengths)]
    #print(eps_threshold)
    feasible = (wall_countour > 0.000001)
    feasible = feasible.flatten()
    while True:
        action_proposed = random.randrange(env.item_count_real * 4)
        if feasible[action_proposed]:
            return action_proposed

GAMMA = 0.95

rewards = []
for i in range(len(env_list)):
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
       # plt.imshow(X)
      #  plt.show()
        # Select and perform an action
        # print(state[0][:,:7])
       # mcts_policy, action, tree_node = mcts.compute_action(tree_node)
        action = select_action(env_list[i])
      #  print(action)
      #  print(mcts_policy)
    #     #  print(action)
     #   env_list[i].set_state(state)
        reward, done = env_list[i].step(action)
        print(reward)

      #  print(out)
        reward_tmp.append(reward)
       # env_list[i].set_state(tree_node.state)

    #     #env_list[i].visualize()
        if done:
            break
    #s    exit(1)
    X = env_list[i].visualize2D()
   # plt.imshow(X)
   # plt.show()
    frames.append(X)
    #print(frames)
    imageio.mimsave('./baseline_random/' + str(i) + '.gif', frames, 'GIF', fps=5, duration=0.075)
    #env_list[i].visualize()
    reward_final = 0
    for x in range(len(reward_tmp) - 1,-1,-1):
        reward_final = (reward_tmp[x] + GAMMA * reward_final)
    rewards.append(reward_final)
    print(reward_tmp)
    print(reward_final)
print(rewards)
