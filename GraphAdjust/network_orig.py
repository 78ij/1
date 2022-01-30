import torch.nn.functional as F
from torch import nn
import random
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNetwork_simple(nn.Module):
    def __init__(self, feature_size, hidden_size, action_space_dim):
        super(DQNetwork_simple, self).__init__()

        self.mlp1 = nn.Linear(feature_size, hidden_size)

        self.LSTM = nn.LSTM(input_size=hidden_size, hidden_size= int(hidden_size / 2))

        self.mlp2 = nn.Linear(int(hidden_size / 2), action_space_dim) 

    def forward(self, state_input,h_state):
       # state_input = self.mlp1(state_input)
        mlp1_out = F.sigmoid(self.mlp1(state_input))
        #print(mlp1_out.shape)
        lstm_out, h_out = self.LSTM(mlp1_out.view(-1,1,2048),h_state)
        out = self.mlp2(F.sigmoid(lstm_out))
        return out,h_out
        #return (out,h,c)
