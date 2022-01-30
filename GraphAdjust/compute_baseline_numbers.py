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
import sys
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
sys.path.append('../')
from scene_graph_structurenet import BasicSceneGraph

print('xxx')
with open('baseline_sync_living.pkl','rb') as f:
    train_data = pickle.load(f)
train_data_size = len(train_data)

tmp_env_list = []
env_list = []
action_list =  []        
for i in range(train_data_size):
    try:
        env_tmp = env_subgraph.ENV(train_data[i])
        env_list.append(env_tmp)
    except: 
        continue
    #X = env_tmp.visualize2D()
    #plt.imshow(X)
    #plt.show()
    env_list.append(env_tmp)
coll = []
affor = []
#robot = []
def get_move_action(dist):
    dist_max = np.argmax(np.abs(dist))

    dist_second = abs(2 - dist_max)

    if dist_max == 0:
        if dist[dist_max]>= 0:
            return 0
        if dist[dist_max]< 0:
            return 1
    if dist_max == 2:
        if dist[dist_max]>= 0:
            return 2
        if dist[dist_max]< 0:
            return 3
    return 0

def rotation_matrix(axis, theta):
    if(np.linalg.norm(axis) < 0.01): return np.identity(4) 
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac),0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab),0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc,0],
                     [0,0,0,1]])
def select_action_random(env):

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
def select_action_heru(env):

    all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wall_countour = env.get_state()
    #print(edge_index)
    length = [len(subbox_lengths)]
    #print(eps_threshold)
    feasible = (wall_countour > 0.000001)
    feasible = feasible.flatten()

    collision_pair = env.getboxcollisionpair()
    wall_coll_idx = env.getwallcollisiondetail()
    print('----------------------------------')
    print(collision_pair)
    if wall_coll_idx is not None:
        wall_vert_center =env.wall_vert_center
        obj = env.data[wall_coll_idx]
        proposed = wall_vert_center - obj[0:3]
        proposed_action = get_move_action(proposed)

        proposed_action = 4 * wall_coll_idx + proposed_action
        if feasible[proposed_action]: return proposed_action

    for pp in collision_pair:

        obj1 = env.data[pp[0]]
        obj2 = env.data[pp[1]]
        extent_obj1 = np.sum(obj1[3:6])
        extent_obj2 = np.sum(obj2[3:6])
        if extent_obj1 >= extent_obj2: move_obj = 1
        else: move_obj = 0
        move_obj = random.choice([0,1])
        selected = pp[move_obj]
        if move_obj == 0:
            proposed = obj1[0:3] - obj2[0:3]
        else:
            proposed = obj2[0:3] - obj1[0:3]
        proposed_action = get_move_action(proposed)

        proposed_action = 4 * selected + proposed_action
        if feasible[proposed_action]: return proposed_action

    moveable_objs = []
    for(k,v) in env.moveable_dict.items():
        moveable_objs.append(k)
    print('----------------------------------')
    print(moveable_objs)
    for k in moveable_objs:
        aaxis = np.array([0,0,1])
        #extent_actual = box.bounding_box.primitive.extents
        rotation = rotation_matrix(np.array([0,1,0]),env.data[k,6])
        ray_dir = np.dot(aaxis,rotation[:3,:3])

        center_other = ray_dir / np.linalg.norm(ray_dir)

        for i in range(env.item_count_real):
            if env.data[i,0] >= center_other[0] - 0.5 and env.data[i,0] <= center_other[0] + 0.5 and env.data[i,2] <= center_other[2] + 0.5 and  env.data[i,2] >= center_other[0] - 0.5:
                obj1 = env.data[i]
                obj2 = env.data[k]
                pp = (i,k)
                extent_obj1 = np.sum(obj1[3:6])
                extent_obj2 = np.sum(obj2[3:6])
                #if extent_obj1 >= extent_obj2: move_obj = 1
               # else: move_obj = 0
                move_obj = random.choice([0,1])
                selected = pp[move_obj]
                if move_obj == 0:
                    proposed = obj1[0:3] - obj2[0:3]
                else:
                    proposed = obj2[0:3] - obj1[0:3]
                proposed_action = get_move_action(proposed)

                proposed_action = 4 * selected + proposed_action
                if feasible[proposed_action]: return proposed_action

    while True:
        action_proposed = random.randrange(env.item_count_real * 4)
        if feasible[action_proposed] != 0:
            return action_proposed
idx = 0
for env in env_list:
    for t in range(30):
        action = select_action_heru(env)
        reward, done = env.step(action)
        if done:
            break
    
    env.output_render_result('./rendered/bad/' + str(idx) + '.png')
    idx += 1
    print(idx)
    coll.append(env.getboxcollision()[0] + env.getwallcollision() * 3)
    affor.append(env.get_free_space())
   # robot.append(env.get_robot_viability())

   # print(coll[-1],affor[-1],robot[-1])
    
coll = np.array(coll)
affor = np.array(affor)
print('----------------------------------------------')
print(coll)
print(affor)
print(np.sum(coll) / len(env_list))
print(np.sum(affor) / len(env_list))