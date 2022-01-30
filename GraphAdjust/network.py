import torch.nn.functional as F
from torch import nn
import torch
import random
from collections import namedtuple
import torch_scatter

from pointnet_model import *

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


class BoxEncoder1(nn.Module):

    def __init__(self, feature_size, hidden_size, orient = False):
        super(BoxEncoder1, self).__init__()

        #self.mlp_skip = nn.Linear(7, feature_size)
        self.mlp1 = nn.Sequential(nn.Linear(7, hidden_size), nn.LeakyReLU(0.1))
        self.mlp2 = nn.Linear(hidden_size, hidden_size)


    def forward(self, box_input):
        net = F.leaky_relu(self.mlp1(box_input), 0.1)
        net = F.leaky_relu(self.mlp2(net), 0.1)
        # box_vector = torch.nn.functional.leaky_relu(self.encoder(box_input), 0.1)
        return net

class MoveableEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size, orient = False):
        super(MoveableEncoder, self).__init__()

        #self.mlp_skip = nn.Linear(8, hidden_size)
        self.mlp1 = nn.Sequential(nn.Linear(8, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, hidden_size))
        self.mlp2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, box_input):
        #print(box_input.shape)
        net = F.leaky_relu(self.mlp1(box_input), 0.1)
        net = F.leaky_relu(self.mlp_skip(box_input) + self.mlp2(net), 0.1)
        # box_vector = torch.nn.functional.leaky_relu(self.encoder(box_input), 0.1)
        return net
class GNNEncoder(nn.Module):

    def __init__(self, node_feat_size, hidden_size, node_symmetric_type, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNEncoder, self).__init__()

        self.boxencoder = BoxEncoder1(node_feat_size,hidden_size)
        self.moveableencoder = MoveableEncoder(node_feat_size,hidden_size)
        self.node_symmetric_type = node_symmetric_type
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num
        self.num_sem = 7#len(category_class)
        self.child_op = nn.Linear(node_feat_size,hidden_size)
        # nn.Linear(node_feat_size * 2+ self.num_sem + 128, hidden_size)
        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*2+edge_type_num, hidden_size))
        self.hidden_size = hidden_size
        self.node_feat_size = node_feat_size
        self.parent_op = nn.Linear(hidden_size*(self.num_iterations+1), node_feat_size)
        self.skip_op_object = nn.Linear(node_feat_size,node_feat_size) #* 2 + self.num_sem + 128, node_feat_size)
        self.second_object = nn.Linear(hidden_size*(self.num_iterations+1), node_feat_size)

    """
        Input Arguments:
            child feats: b x max_childs x feat_dim
            child exists: b x max_childs x 1
            edge_type_onehot: b x num_edges x edge_type_num
            edge_indices: b x num_edges x 2
    """
    def forward(self, child_feats, edge_indices,edge_type_onehot, lengths):
        #print(child_feats.shape)
        #print(edge_indices)
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        num_edges = edge_indices.shape[1]
        hidden_size = self.hidden_size
        node_feat_size = self.node_feat_size
        #child_feats[:,:,:3] *= 100
       # print(child_feats)
        if batch_size != 1:
            raise ValueError('Currently only a single batch is supported.')
        child_feats_encoded = self.boxencoder(child_feats[:,:,:7])
       # moveable_feats_encoded = self.moveableencoder(child_feats[:,:,142:])
       # child_feats = torch.cat([child_feats_encoded,child_feats[:,:,7:142],moveable_feats_encoded],dim=2)
        #skip_feats = self.skip_op_object(child_feats).reshape(-1,node_feat_size)
        child_feats = child_feats_encoded
        # perform MLP for child features
       
        #child_feats = torch.relu(self.child_op(child_feats))

        # zero out non-existent children
        #child_feats = child_feats * child_exists
        #child_feats = child_feats.view(1, max_childs, -1)
        #skip_feats = skip_feats * child_exists
        #skip_feats = skip_feats.view(1, max_childs, -1)

        # combine node features before and after message-passing into one parent feature
        iter_parent_feats = []
        iter_parent_feats.append(child_feats.reshape(-1,hidden_size))
       # skip_feat = F.leaky_relu(skip_feats, 0.1)

        # if self.node_symmetric_type == 'max':
        #     child_feats_tmp = list(child_feats.reshape(-1,hidden_size).split(lengths,dim=0))
        #     child_feats_tmp = [e.max(dim=0)[0].reshape(1,hidden_size) for e in child_feats_tmp]
        #     child_feats_tmp = torch.cat(child_feats_tmp,dim=0)


        #     #print(skip_feats.reshape(-1,hidden_size).shape)
        #     #print(lengths)
        #     skip_feats = list(skip_feats.reshape(-1,node_feat_size).split(lengths,dim=0))
        #     skip_feats = [e.max(dim=0)[0].reshape(1,node_feat_size) for e in skip_feats]

        #     skip_feats = torch.cat(skip_feats,dim=0)
           
        # elif self.node_symmetric_type == 'sum':
        #     iter_parent_feats.append(child_feats.sum(dim=1))
        #     skip_feat = F.leaky_relu(skip_feats.max(dim=1)[0], 0.1)
        # elif self.node_symmetric_type == 'avg':
        #     iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
        #     skip_feat = F.leaky_relu(skip_feats.max(dim=1)[0], 0.1)
        # else:
        #     raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        if self.num_iterations > 0 and num_edges > 0:
            edge_feats = edge_type_onehot

        edge_indices_from = edge_indices[:, :, 0].view(-1, 1).expand(-1, hidden_size)
        #print(edge_indices)
        # perform Graph Neural Network for message-passing among sibling nodes
        for i in range(self.num_iterations):
            if num_edges > 0:
                # MLP for edge features concatenated with adjacent node features
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[0, :, 0], :], # start node features
                    child_feats[0:1, edge_indices[0, :, 1], :], # end node features
                    edge_feats], dim=2) # edge features

                #print(node_edge_feats.shape)
                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))
                node_edge_feats = node_edge_feats.view(num_edges, -1)

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                #child_feats = new_child_feats.view(1, max_childs, hidden_size)
                child_feats = new_child_feats.view(1,-1,hidden_size)
                #child_feats_tmp = new_child_feats.view(-1,hidden_size)
                #child_feats_tmp = list(child_feats_tmp.split(lengths,dim=0))
            # combine node features before and after message-passing into one parent feature
           # print(child_feats)
            iter_parent_feats.append(child_feats.view(-1,hidden_size))
            # if self.node_symmetric_type == 'max':
            #     child_feats_tmp = [e.max(dim=0)[0].reshape(1,hidden_size) for e in child_feats_tmp]
            #     child_feats_tmp = torch.cat(child_feats_tmp,dim=0)
            #     iter_parent_feats.append(child_feats_tmp)
            #     #iter_parent_feats.append(child_feats.max(dim=1)[0])
            # elif self.node_symmetric_type == 'sum':
            #     iter_parent_feats.append(child_feats.sum(dim=1))
            # elif self.node_symmetric_type == 'avg':
            #     iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            # else:
            #     raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        # concatenation of the parent features from all iterations (as in GIN, like skip connections)
        parent_feat = torch.cat(iter_parent_feats, dim=1).reshape(-1,hidden_size*(self.num_iterations+1))
        #print(skip_feat.shape)

        # back to standard feature space size
        parent_feat = F.leaky_relu(self.second_object(parent_feat), 0.1)
        #print(parent_feat)
        return parent_feat

class DQNetwork_simple(nn.Module):
    def __init__(self, hidden_size, action_space_dim):
        super(DQNetwork_simple, self).__init__()

        self.gnnencoder = GNNEncoder(256, 512,'max','max',2,3)

        self.mlp1 = nn.Linear(256, hidden_size)
        #self.LSTM = nn.LSTM(input_size=hidden_size, hidden_size=128)

        self.mlpm = nn.Linear(hidden_size, int(hidden_size / 2))

        self.mlp2 = nn.Linear(int(hidden_size / 2), 4) 

        #self.pointnet = PointNetfeat(global_feat=True)

    def forward(self, data_input,edge_index,edge_type,lengths):
        #print(state_input)

        graph_feature = self.gnnencoder(data_input,edge_index,edge_type,lengths)
        

        state_input = self.mlp1(graph_feature)
       # lstm_out, (h,c) = self.LSTM(state_input.view(1,1,-1))

       # print(state_input.shape)
        #print(state_input.shape)
        #pn_out = self.pointnet(state_input)[0]
      
        out = torch.tanh(self.mlp2(torch.sigmoid(self.mlpm(torch.sigmoid(state_input))))) * 50
        #print(edge_index.shape)
        edge_from = edge_index[0,:,0].flatten()
        edge_to = edge_index[0,:,1].flatten()

        node_from = out[edge_from,:]
        node_to = out[edge_to,:]

       # loss_diff = torch.mean(torch.exp(-torch.abs(node_from - node_to)))
        loss_diff = 0
        #print(out.shape)
        #print(edge_index)
        out = list(out.split(lengths,dim=0))


        out_final = torch.ones((len(out),80),device='cuda:0') * -10000000
        for i in range(len(out)):
            out_final[i,0:4*out[i].shape[0]] = out[i].flatten()
        return out_final,loss_diff
        #return (out,h,c)
