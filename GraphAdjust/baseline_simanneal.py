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
sys.path.append('../')
from scene_graph_structurenet import BasicSceneGraph
print('xxx')
with open('baseline_sync_living.pkl','rb') as f:
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
    except:
        continue
val_env_list = env_list#[500:]
train_data_size = len(env_list)


_temperature = 10
# void Sampler::anneal(int iteration) {
#     if (iteration % 100 == 0){
#         _temperature *= 0.9;
#     }
#     _stepSize = (_iterationMax-iteration)/double(_iterationMax) + 0.01;
# }
# 
_iterationMax = 0
iteration = 0
stepSize = 0
def anneal(i):
    global _temperature
    global stepSize
    global _iterationMax
    if i % 100 == 0: _temperature *= 0.9
    stepSize = (_iterationMax-i)/float(_iterationMax) / 20
#void Sampler::run_mcmc(Room &room, size_t startFurnIndex, size_t endFurnIndex) {
#         _iterationMax = 5000 * room.get_furniture_list().size();
#         //std::cout << "compute_total_energy" << std::endl;
#         double currentEnergy = room.compute_total_energy(_costWeights), proposedEnergy;

#         _temperature = 10;
#         for (int i = 0; i < _iterationMax; i++) {
#             //std::cout << "MCMC iteration: " << i << ", temperature: " << _temperature << std::endl;
#             anneal(i);

#             Room proposedRoom = room;
#             proposedRoom.propose(_stepSize, startFurnIndex, endFurnIndex, false);
#             proposedEnergy = proposedRoom.compute_total_energy(_costWeights);

#             double r = (rand() / (double)RAND_MAX), acceptanceProb = std::min(1.0, exp(  (currentEnergy - proposedEnergy)/_temperature) );
#             if (r < acceptanceProb){
#                 //std::cout << "Accepted: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
#                 room = proposedRoom;
#                 currentEnergy = proposedEnergy;
#             }else{
#                 //std::cout << "Rejected: " << currentEnergy << " " << proposedEnergy <<  " " << r <<  " " << acceptanceProb <<  " " << _temperature << std::endl;
#             }

#             //if (i % 100 == 0){
#             //    // Output every result to file for visualization
#             //    std::stringstream outputFilename;
#             //    outputFilename << "sample_";
#             //    outputFilename << std::setfill('0') << std::setw(8) << i;
#             //    room.write_to_room_arranger_file(_workspacePath+"tmp/samples/mcmc/txt/"+outputFilename.str()+".txt");
#             //    room.write_to_json(_workspacePath+"tmp/samples/mcmc/json/"+outputFilename.str()+".json");
#             //}
#         }
#     }
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
def rev(last_action):
    ret = 0
    item = int(last_action / 4)
    action = last_action % 4
    if action == 0: ret = 1
    if action == 1: ret = 0
    if action == 2: ret = 3
    if action == 3: ret = 2
    return ret + item * 4
rewards = []
for i in range(len(env_list) - 60):
    _temperature = 10
    frames = []
    coll = []
    affor = []
    env_list[i].reset()
    #env_list[i].visualize()
    reward_tmp = []
    #X = env_list[i].visualize2D()
   # frames.append(X)
    #plt.imshow(X)
   # plt.show()
    _iterationMax = 3000
    for t in range(4000):
        print('iter ' + str(t))
        anneal(t)
      #  plt.imshow(X)
    #    plt.show()
       # plt.imshow(X)
      #  plt.show()
        # Select and perform an action
        # print(state[0][:,:7])
       # mcts_policy, action, tree_node = mcts.compute_action(tree_node)
        #action = select_action(env_list[i])
      #  print(action)
      #  print(mcts_policy)
    #     #  print(action)
     #   env_list[i].set_state(state)
        #reward, done = env_list[i].step(action)
        idx = random.randint(0,env_list[i].item_count_real)
        translation = np.array([0.,0.])
        print('stepsize ' + str(stepSize))
        translation[0] = stepSize*2*(random.random() - 0.5)
        translation[1] = stepSize*2*(random.random() - 0.5)
        print(idx)
        print(translation)
        reward, done = env_list[i].step_random(idx,translation)
        if done: 
            env_list[i].step_random(idx,-translation)
            continue

        print('reward ' +str(reward))
        print('temperature ' + str(_temperature))
        print(reward)
        r = random.random()
        acceptanceProb = min(1.0, math.exp( reward/_temperature / 5))
        if r > acceptanceProb:
            reward, done = env_list[i].step_random(idx,-translation)
            print('not accepted')
            if len(frames) * 300 <= t:
                print('append!')
        else:
            print('accepted')
            if len(frames) * 300 <= t:
                print('append!')
                #X = env_list[i].visualize2D()
                #frames.append(X)
    coll.append(env_list[i].getboxcollision()[0] + env_list[i].getwallcollision() * 3)
    affor.append(env_list[i].get_free_space())
   # robot.append(env.get_robot_viability())

   # print(coll[-1],affor[-1],robot[-1])
    
coll = np.array(coll)
affor = np.array(affor)
print('----------------------------------------------')
print(coll)
print(affor)
print(np.sum(coll) / len(env_list))
print(np.sum(affor) / len(env_list))
                
      #  print(out)
            #reward_tmp.append(reward)
       # env_list[i].set_state(tree_node.state)

    #     #env_list[i].visualize()
        # if done:
        #     break
    #s    exit(1)
   # plt.imshow(X)
   # plt.show()
    #print(frames)
   # imageio.mimsave('./baseline_simanneal/' + str(i) + '.gif', frames, 'GIF', fps=5, duration=0.075)
    #env_list[i].visualize()
#     reward_final = 0
#     for x in range(len(reward_tmp) - 1,-1,-1):
#         reward_final = (reward_tmp[x] + GAMMA * reward_final)
#     rewards.append(reward_final)
#     print(reward_tmp)
#     print(reward_final)
# print(rewards)
