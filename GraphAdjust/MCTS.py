"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt


class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env
        self.reward = 0
        self.depth=0
class Node:
    def __init__(self, action, done, reward, state, mcts, parent=None,depth=0):
        self.env = parent.env
        self.depth=depth
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = 80
        self.child_total_value = np.ones(
            [self.action_space_size], dtype=np.float32) * -500  # Q
        self.child_priors = np.zeros(
            [self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)  # N
        self.valid_actions = np.zeros(
            [self.action_space_size], dtype=np.bool)  # N
        self.done_actions = np.zeros(
            [self.action_space_size], dtype=np.bool)  # N
        self.valid_actions[:self.env.item_count_real*5] = True

        self.reward = reward
        self.done = done
        self.state = state
       # self.obs = obs

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        value_2 = self.child_total_value[:self.env.item_count_real*5]
        child_total_value_normlized = np.zeros_like(self.child_total_value)
        child_total_value_normlized[:self.env.item_count_real*5] = (value_2 - np.min(value_2) + 1) / (np.max(value_2) - np.min(value_2) + 1)
      #  print(child_total_value_normlized)
        return child_total_value_normlized
    def child_U(self):
        return np.sqrt(np.log(self.number_visits)   / (
            1 + self.child_number_visits))

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions | self.done_actions] = -np.inf
       # print(masked_child_score)
        return np.argmax(masked_child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node_tmp,is_valid = current_node.get_child(best_action)
            if is_valid: current_node = current_node_tmp
          #  print(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state(self.state)
           # X =self.env.visualize2D()
           # plt.imshow(X)
            #plt.show()
            reward, done = self.env.step(action)
            next_state = self.env.get_state_for_mcts()
          #  X =self.env.visualize2D()
          #  plt.imshow(X)
          #  plt.show()
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                mcts=self.mcts,depth=self.depth+1)
            if done:
                self.child_total_value[action] = reward
                self.done_actions[action] = True
        if self.done_actions[action]:
            return self.children[action],False
        else:
            return self.children[action],True

    def backup(self):
        current = self
        value = 0
        while current.parent is not None:
           # print('depth: ' + str(current.depth))
            current = current.parent
            if isinstance(current,RootParentNode): break
            action_max = np.argmax(current.child_total_value)
            self.env.set_state(self.state)
            reward, done = self.env.step(action_max)
            current.total_value = 0.8 * self.child_total_value[action_max] + reward
            current.number_visits += 1

          #  value *= 0.8




class MCTS:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
    def sim(self,leaf):
        best_action = np.argmax(leaf.child_priors)
        state_tmp = leaf.state
        s = 0
        leaf.env.set_state(state_tmp) 
        if leaf.env.isdone():
            leaf.number_visits = 1
            leaf.total_value = 5
            leaf.backup()
            return
        gamma = 0.8
        for i in range(20):
            
            leaf.env.set_state(state_tmp)
           # X =self.env.visualize2D()
           # plt.imshow(X)
            #plt.show()
            reward, done = leaf.env.step(best_action)
            #print(reward)
            s += reward * gamma
            gamma *= 0.8
            if done: break
            state_tmp = leaf.env.get_state_for_mcts()

            _,all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wallcountour = state_tmp
    #print(edge_index)
            length = [len(subbox_lengths)]
            all_edge = torch.tensor(all_edge.astype("long"),device='cuda:0')
            all_type = torch.tensor(all_type.astype("float32"),device='cuda:0')
            data = torch.tensor(data.astype("float32"),device='cuda:0')
            edge_index = torch.tensor(edge_index.astype("long"),device='cuda:0')
            edge_type = torch.tensor(edge_type.astype("float32"),device='cuda:0')
            wallcountour = torch.tensor(wallcountour.astype("float32"),device='cuda:0')


            child_priors,_ = self.model(all_edge.reshape(1,-1,2), all_type.reshape(1,-1,3),data.reshape(1,-1,11), edge_index.long().reshape(1,-1,2), edge_type.reshape(1,-1,3),length,subbox_lengths,wallcountour)
            child_priors = scipy.special.softmax(child_priors.detach().cpu().numpy().flatten())
            best_action = np.argmax(leaf.child_priors)
      #  print(s)
        leaf.number_visits = 1
        leaf.total_value = s
        leaf.backup()


    def compute_action(self, node):
        for x in range(self.num_sims):
            leaf = node.select()
           # value = leaf.reward
            _,all_edge,all_type,data,edge_index,edge_type,subbox_lengths,wallcountour = leaf.state
    #print(edge_index)
            length = [len(subbox_lengths)]
            all_edge = torch.tensor(all_edge.astype("long"),device='cuda:0')
            all_type = torch.tensor(all_type.astype("float32"),device='cuda:0')
            data = torch.tensor(data.astype("float32"),device='cuda:0')
            edge_index = torch.tensor(edge_index.astype("long"),device='cuda:0')
            edge_type = torch.tensor(edge_type.astype("float32"),device='cuda:0')
            wallcountour = torch.tensor(wallcountour.astype("float32"),device='cuda:0')


            child_priors,_ = self.model(all_edge.reshape(1,-1,2), all_type.reshape(1,-1,3),data.reshape(1,-1,11), edge_index.long().reshape(1,-1,2), edge_type.reshape(1,-1,3),length,subbox_lengths,wallcountour)
            child_priors = scipy.special.softmax(child_priors.detach().cpu().numpy().flatten())
           # print(str(x) + '   ' + str(child_priors))
            if self.add_dirichlet_noise:
                child_priors = (1 - self.dir_epsilon) * child_priors
                child_priors += self.dir_epsilon * np.random.dirichlet(
                    [self.dir_noise] * child_priors.size)

            leaf.expand(child_priors)
           # leaf.backup(value)
            self.sim(leaf)
        # Tree policy target (TPT)
        tree_policy = node.child_total_value
       # tree_policy = tree_policy / np.max(
       #     tree_policy)  # to avoid overflows when computing softmax
        tree_policy = (tree_policy - np.min(tree_policy) + 1) / (np.max(tree_policy) - np.min(tree_policy) + 1)
      #  tree_policy = np.power(tree_policy, self.temperature)
     #   tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(
                np.arange(node.action_space_size), p=tree_policy)
        print('total')
        print(node.child_total_value)
        print('prior')
        print(node.child_priors)
        leaf.env.set_state(node.children[action].state)
       # X =leaf.env.visualize2D()
      #  plt.imshow(X)
      #  plt.show()
        return tree_policy, action, node.children[action]