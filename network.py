import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_scatter
from common import *
from scipy.optimize import linear_sum_assignment
from utils import linear_assignment, load_pts, transform_pc_batch, get_surface_reweighting_batch
from chamfer_distance import ChamferDistance

class BoxEncoder1(nn.Module):

    def __init__(self, feature_size, hidden_size, orient = False):
        super(BoxEncoder1, self).__init__()

        self.mlp_skip = nn.Linear(10, feature_size)
        self.mlp1 = nn.Sequential(nn.Linear(10, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, hidden_size))
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, box_input):
        net = F.leaky_relu(self.mlp1(box_input), 0.1)
        net = F.leaky_relu(self.mlp_skip(box_input) + self.mlp2(net), 0.1)
        # box_vector = torch.nn.functional.leaky_relu(self.encoder(box_input), 0.1)
        return net

class BoxEncoder(nn.Module):

    def __init__(self, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(10, feature_size)

    def forward(self, box_input):
        box_vector = torch.relu(self.encoder(box_input))
        return box_vector

class BoxDecoder1(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDecoder1, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, hidden_size))
        self.center = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, 3))
        self.size = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, 3))
        self.rot = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, 4))
        self.center_skip = nn.Linear(feature_size, 3)
        self.size_skip = nn.Linear(feature_size, 3)
        self.rot_skip = nn.Linear(feature_size, 4)


    def forward(self, parent_feature):
        feat = torch.nn.functional.leaky_relu(self.mlp(parent_feature), 0.1)
        center = torch.tanh(self.center(feat) + self.center_skip(parent_feature)) 
        # size = torch.sigmoid(self.size(feat)) * 22
        # size = torch.abs(self.size(feat) + self.size_skip(parent_feature))
        size = torch.sigmoid(self.size_skip(parent_feature)+self.size(feat))
        # size = torch.nn.functional.leaky_relu(self.size(feat),0.1)

        rot_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]]) 
        rot = torch.tanh(self.rot_skip(parent_feature) + self.rot(feat)).add(rot_bias)
        rot = rot / (1e-12 + rot.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        vector = torch.cat([center, size, rot], dim=1)
        return vector

class BoxDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.center = nn.Linear(hidden_size, 3)
        self.size = nn.Linear(hidden_size, 3)
        self.quat = nn.Linear(hidden_size, 4)

    def forward(self, parent_feature):
        feat = torch.relu(self.mlp(parent_feature))
        center = torch.tanh(self.center(feat))
        size = torch.sigmoid(self.size(feat)) * 2
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        vector = torch.cat([center, size, quat], dim=1)
        return vector

class GNNEncoderStructureNet(nn.Module):

    def __init__(self, node_feat_size, hidden_size, node_symmetric_type, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNEncoderStructureNet, self).__init__()

        self.boxencoder = BoxEncoder1(node_feat_size,hidden_size)
        self.node_symmetric_type = node_symmetric_type
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
       # self.edge_type_num = edge_type_num
        self.num_sem = len(category_class)
        self.child_op = nn.Linear(node_feat_size + self.num_sem, hidden_size)
        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*2, hidden_size))

        self.parent_op = nn.Linear(hidden_size*(self.num_iterations+1), node_feat_size)
        self.skip_op_object = nn.Linear(node_feat_size + self.num_sem, node_feat_size)
        self.second_object = nn.Linear(hidden_size*(self.num_iterations+1), node_feat_size)

    """
        Input Arguments:
            child feats: b x max_childs x feat_dim
            child exists: b x max_childs x 1
            edge_type_onehot: b x num_edges x edge_type_num
            edge_indices: b x num_edges x 2
    """
    def forward(self, child_feats, child_exists, edge_indices):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        num_edges = edge_indices.shape[1]
       # print(child_feats)
        if batch_size != 1:
            raise ValueError('Currently only a single batch is supported.')

        child_feats_encoded = self.boxencoder(child_feats[:,:,:10])
        child_feats = torch.cat([child_feats_encoded,child_feats[:,:,10:]],dim=2)
        # perform MLP for child features
        skip_feats = self.skip_op_object(child_feats)
        child_feats = torch.relu(self.child_op(child_feats))
        hidden_size = child_feats.size(-1)

        # zero out non-existent children
        child_feats = child_feats * child_exists
        child_feats = child_feats.view(1, max_childs, -1)
        skip_feats = skip_feats * child_exists
        skip_feats = skip_feats.view(1, max_childs, -1)

        # combine node features before and after message-passing into one parent feature
        iter_parent_feats = []
        if self.node_symmetric_type == 'max':
            iter_parent_feats.append(child_feats.max(dim=1)[0])
            skip_feat = F.leaky_relu(skip_feats.max(dim=1)[0], 0.1)
        elif self.node_symmetric_type == 'sum':
            iter_parent_feats.append(child_feats.sum(dim=1))
            skip_feat = F.leaky_relu(skip_feats.max(dim=1)[0], 0.1)
        elif self.node_symmetric_type == 'avg':
            iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            skip_feat = F.leaky_relu(skip_feats.max(dim=1)[0], 0.1)
        else:
            raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

      # if self.num_iterations > 0 and num_edges > 0:
       #     edge_feats = edge_type_onehot

        edge_indices_from = edge_indices[:, :, 0].view(-1, 1).expand(-1, hidden_size)
        #print(edge_indices)
        # perform Graph Neural Network for message-passing among sibling nodes
        for i in range(self.num_iterations):
            if num_edges > 0:
                # MLP for edge features concatenated with adjacent node features
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[0, :, 0], :], # start node features
                    child_feats[0:1, edge_indices[0, :, 1], :], # end node features
                    ], dim=2) # edge features

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

                child_feats = new_child_feats.view(1, max_childs, hidden_size)

            # combine node features before and after message-passing into one parent feature
            if self.node_symmetric_type == 'max':
                iter_parent_feats.append(child_feats.max(dim=1)[0])
            elif self.node_symmetric_type == 'sum':
                iter_parent_feats.append(child_feats.sum(dim=1))
            elif self.node_symmetric_type == 'avg':
                iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            else:
                raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        # concatenation of the parent features from all iterations (as in GIN, like skip connections)
        parent_feat = torch.cat(iter_parent_feats, dim=1)

        # back to standard feature space size
        parent_feat = F.leaky_relu(skip_feat + self.second_object(parent_feat), 0.1)

        return parent_feat



class GNNDecoderStructureNet(nn.Module):

    def __init__(self, node_feat_size, hidden_size, max_child_num, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNDecoderStructureNet, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num
        self.num_sem = len(category_class)
        
        self.box_decoder = BoxDecoder(node_feat_size,hidden_size)
        self.mlp_parent = nn.Linear(node_feat_size, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, self.num_sem)
        self.mlp_child = nn.Linear(hidden_size, node_feat_size)
        self.mlp_edge_latent = nn.Linear(hidden_size*2, hidden_size)

        self.mlp_edge_exists = nn.ModuleList()
        for i in range(edge_type_num):
            self.mlp_edge_exists.append(nn.Linear(hidden_size, 1))

        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*2, hidden_size))

        self.mlp_child = nn.Linear(hidden_size*(self.num_iterations+1), hidden_size)
        self.mlp_sem = nn.Linear(hidden_size, self.num_sem)
        self.mlp_child2 = nn.Linear(hidden_size, node_feat_size)
        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))
        self.register_buffer('anchor', torch.from_numpy(load_pts('anchor.pts')))
        self.chamferLoss = ChamferDistance()
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.mseLoss = nn.SmoothL1Loss(reduction='none')

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        if batch_size != 1:
            raise ValueError('Only batch size 1 supported for now.')

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)
        #print(child_feats.size())
        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size*self.max_child_num, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # edge features
        edge_latents = torch.cat([
            child_feats.view(batch_size, self.max_child_num, 1, self.hidden_size).expand(-1, -1, self.max_child_num, -1),
            child_feats.view(batch_size, 1, self.max_child_num, self.hidden_size).expand(-1, self.max_child_num, -1, -1)
            ], dim=3)
        edge_latents = torch.relu(self.mlp_edge_latent(edge_latents))

        # edge existence prediction
        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(\
                    batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)

        """
            decoding stage message passing
            there are several possible versions, this is a simple one:
            use a fixed set of edges, consisting of existing edges connecting existing nodes
            this set of edges does not change during iterations
            iteratively update the child latent features
            then use these child latent features to compute child features and semantics
        """
        # get edges that exist between nodes that exist
        edge_indices = torch.nonzero(edge_exists_logits > 0)
        edge_types = edge_indices[:, 3]
        edge_indices = edge_indices[:, 1:3]
        nodes_exist_mask = (child_exists_logits[0, edge_indices[:, 0], 0] > 0) \
                & (child_exists_logits[0, edge_indices[:, 1], 0] > 0)
        edge_indices = edge_indices[nodes_exist_mask, :]
        edge_types = edge_types[nodes_exist_mask]

        # get latent features for the edges
       # edge_feats_mp = edge_latents[0:1, edge_indices[:, 0], edge_indices[:, 1], :]

        # append edge type to edge features, so the network has information which
        # # of the possibly multiple edges between two nodes it is working with
        # edge_type_logit = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], :]
        # edge_type_logit = edge_feats_mp.new_zeros(edge_feats_mp.shape[:2]+(self.edge_type_num,))
        # edge_type_logit[0:1, range(edge_type_logit.shape[1]), edge_types] = \
        #         edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], edge_types]
        # edge_feats_mp = torch.cat([edge_feats_mp, edge_type_logit], dim=2)

        num_edges = edge_indices.shape[0]
        max_childs = child_feats.shape[1]

        iter_child_feats = [child_feats] # zeroth iteration

        if self.num_iterations > 0 and num_edges > 0:
            edge_indices_from = edge_indices[:, 0].view(-1, 1).expand(-1, self.hidden_size)

        for i in range(self.num_iterations):
            if num_edges > 0:
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[:, 0], :], # start node features
                    child_feats[0:1, edge_indices[:, 1], :], # end node features
                    ], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, self.hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, self.hidden_size)

            # save child features of this iteration
            iter_child_feats.append(child_feats)

        # concatenation of the child features from all iterations (as in GIN, like skip connections)
        child_feats = torch.cat(iter_child_feats, dim=2)

        # transform concatenation back to original feature space size
        child_feats = child_feats.view(-1, self.hidden_size*(self.num_iterations+1))
        child_feats = torch.relu(self.mlp_child(child_feats))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.hidden_size)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, self.num_sem)

        # node features
        child_feats = self.mlp_child2(child_feats.view(-1, self.hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        return child_feats, child_sem_logits, child_exists_logits, edge_exists_logits

    def anchorLossEstimator(self, box_feature, gt_box_feature):
        pred_anchor_pc = transform_pc_batch(self.anchor, box_feature, anchor=True)
        gt_anchor_pc = transform_pc_batch(self.anchor, gt_box_feature, anchor=True)
        dist1, dist2 = self.chamferLoss(gt_anchor_pc, pred_anchor_pc)
        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        return loss
    def boxLossEstimator(self, box_feature, gt_box_feature):
        pred_box_pc = transform_pc_batch(self.unit_cube, box_feature)
        with torch.no_grad():
            pred_reweight = get_surface_reweighting_batch(box_feature[:, 3:6], self.unit_cube.size(0))
        gt_box_pc = transform_pc_batch(self.unit_cube, gt_box_feature)
        with torch.no_grad():
            gt_reweight = get_surface_reweighting_batch(gt_box_feature[:, 3:6], self.unit_cube.size(0))
        dist1, dist2 = self.chamferLoss(gt_box_pc, pred_box_pc)
        loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
        loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
        loss = (loss1 + loss2) / 2
        return loss
     # compute per-node loss + children relationship loss
    def graph_recon_loss(self, node_latent, gt_data, gt_edge_index):
        gt_data = gt_data.reshape(1,-1,16)
        gt_edge_index = gt_edge_index.reshape(1,-1,2)
        #print(gt_edge_index)

        child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
                self.forward(node_latent)

        # generate box prediction for each child
        feature_len = node_latent.size(1)
        child_pred_boxes = self.box_decoder(child_feats.view(-1, feature_len))
        num_child_parts = child_pred_boxes.size(0)

        # perform hungarian matching between pred boxes and gt boxes
        with torch.no_grad():
            child_gt_boxes = gt_data[0, :,:10]
            num_gt = child_gt_boxes.size(0)
            child_pred_boxes_tiled = child_pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
            child_gt_boxes_tiled = child_gt_boxes.unsqueeze(dim=1).repeat(1, num_child_parts, 1)

            dist_mat = self.boxLossEstimator(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_child_parts)
            #dist_mat = torch.sum(self.mseLoss(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)), dim = 1).view(-1, num_gt, num_child_parts)
            _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

            # get edge ground truth
            #edge_type_list_gt, edge_indices_gt = gt_node.edge_tensors(
            #    edge_types=self.conf.edge_types, device=child_feats.device, type_onehot=False)
            edge_indices_gt = gt_edge_index
           # print(edge_indices_gt.shape)
            gt2pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(matched_gt_idx, matched_pred_idx)}
            edge_exists_gt = torch.zeros_like(edge_exists_logits)
          #  sym_from = []; sym_to = []; sym_mat = []; sym_type = []; adj_from = []; adj_to = [];
            for i in range(edge_indices_gt.shape[1]//2):
                if edge_indices_gt[0, i, 0].item() not in gt2pred or edge_indices_gt[0, i, 1].item() not in gt2pred:
                    """
                        one of the adjacent nodes of the current gt edge was not matched 
                        to any node in the prediction, ignore this edge
                    """
                    continue

               # print(edge_exists_logits)
                # correlate gt edges to pred edges
                edge_from_idx = gt2pred[edge_indices_gt[0, i, 0].item()]
                edge_to_idx = gt2pred[edge_indices_gt[0, i, 1].item()]
                edge_exists_gt[:, edge_from_idx, edge_to_idx] = 1
                edge_exists_gt[:, edge_to_idx, edge_from_idx] = 1

                # # compute binary edge parameters for each matched pred edge
                # if edge_type_list_gt[0, i].item() == 0: # ADJ
                #     adj_from.append(edge_from_idx)
                #     adj_to.append(edge_to_idx)
                # else:   # SYM
                #     if edge_type_list_gt[0, i].item() == 1: # ROT_SYM
                #         mat1to2, mat2to1 = compute_sym.compute_rot_sym(child_pred_boxes[edge_from_idx].cpu().detach().numpy(), child_pred_boxes[edge_to_idx].cpu().detach().numpy())
                #     elif edge_type_list_gt[0, i].item() == 2: # TRANS_SYM
                #         mat1to2, mat2to1 = compute_sym.compute_trans_sym(child_pred_boxes[edge_from_idx].cpu().detach().numpy(), child_pred_boxes[edge_to_idx].cpu().detach().numpy())
                #     else:   # REF_SYM
                #         mat1to2, mat2to1 = compute_sym.compute_ref_sym(child_pred_boxes[edge_from_idx].cpu().detach().numpy(), child_pred_boxes[edge_to_idx].cpu().detach().numpy())
                #     sym_from.append(edge_from_idx)
                #     sym_to.append(edge_to_idx)
                #     sym_mat.append(torch.tensor(mat1to2, dtype=torch.float32, device=self.conf.device).view(1, 3, 4))
                #     sym_type.append(edge_type_list_gt[0, i].item())

        # # train the current node to be non-leaf
        # is_leaf_logit = self.leaf_classifier(node_latent)
        # is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

        # train all node box to gt
        box_loss = 0
        anchor_loss = 0
        for i in range(len(matched_gt_idx)):
            box_loss += self.boxLossEstimator(gt_data[0,matched_gt_idx[i],:10].reshape(1,-1),child_pred_boxes[matched_pred_idx[i],:].reshape(1,-1))
            #box_loss += self.mseLoss(gt_data[0,matched_gt_idx[i],:10].reshape(1,-1),child_pred_boxes[matched_pred_idx[i],:].reshape(1,-1)).mean()
            anchor_loss = self.anchorLossEstimator(gt_data[0,matched_gt_idx[i],:10].reshape(1,-1),child_pred_boxes[matched_pred_idx[i],:].reshape(1,-1))
        #box_loss /= 20
        #anchor_loss /= 20
        #print(gt_data)
        #print(child_pred_boxes)
        #print(gt2pred)

        # gather information
        child_sem_gt_labels = []
        child_sem_pred_logits = []
        child_box_gt = []
        child_box_pred = []
        child_exists_gt = torch.zeros_like(child_exists_logits)
        #(matched_pred_idx)
        #print(matched_gt_idx)
        for i in range(len(matched_gt_idx)):

            gt_label_tmp = gt_data[0,matched_gt_idx[i],10:]
            # cope with 'no category' circumstance
            #if torch.sum(gt_label_tmp) == 0:
            #    gt_label_tmp[0] = 1
            child_sem_gt_labels.append(torch.where(gt_label_tmp == 1)[0])
            child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
            child_exists_gt[:, matched_pred_idx[i], :] = 1

        # train semantic labels
        child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
        
        child_sem_gt_labels = torch.cat(child_sem_gt_labels, dim=0)
        child_sem_gt_labels = child_sem_gt_labels.long()
       # print(child_sem_gt_labels)
        semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
        semantic_loss = semantic_loss.sum()

        # train unused boxes to zeros
        unmatched_boxes = []
        for i in range(num_child_parts):
            if i not in matched_pred_idx:
                unmatched_boxes.append(child_pred_boxes[i, 0:6].view(1, -1))
        if len(unmatched_boxes) > 0:
            unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
            unused_box_loss = unmatched_boxes.pow(2).sum() * 10
        else:
            unused_box_loss = 0.0

        # train exist scores
        child_exists_loss = F.binary_cross_entropy_with_logits(\
            input=child_exists_logits, target=child_exists_gt, reduction='none')
        child_exists_loss = child_exists_loss.sum()

        # train edge exists scores
        edge_exists_loss = F.binary_cross_entropy_with_logits(\
                input=edge_exists_logits, target=edge_exists_gt, reduction='none')
        edge_exists_loss = edge_exists_loss.sum()
        # rescale to make it comparable to other losses, 
        # which are in the order of the number of child nodes
        edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2]*edge_exists_gt.shape[3]) 

        # compute and train binary losses
        # sym_loss = 0
        # if len(sym_from) > 0:
        #     sym_from_th = torch.tensor(sym_from, dtype=torch.long, device=self.conf.device)
        #     obb_from = child_pred_boxes[sym_from_th, :]
        #     with torch.no_grad():
        #         reweight_from = get_surface_reweighting_batch(obb_from[:, 3:6], self.unit_cube.size(0))
        #     pc_from = transform_pc_batch(self.unit_cube, obb_from)
        #     sym_to_th = torch.tensor(sym_to, dtype=torch.long, device=self.conf.device)
        #     obb_to = child_pred_boxes[sym_to_th, :]
        #     with torch.no_grad():
        #         reweight_to = get_surface_reweighting_batch(obb_to[:, 3:6], self.unit_cube.size(0))
        #     pc_to = transform_pc_batch(self.unit_cube, obb_to)
        #     sym_mat_th = torch.cat(sym_mat, dim=0)
        #     transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat_th[:, :, :3], 1, 2)) + \
        #             sym_mat_th[:, :, 3].unsqueeze(dim=1).repeat(1, pc_from.size(1), 1)
        #     dist1, dist2 = self.chamferLoss(transformed_pc_from, pc_to)
        #     loss1 = (dist1 * reweight_from).sum(dim=1) / (reweight_from.sum(dim=1) + 1e-12)
        #     loss2 = (dist2 * reweight_to).sum(dim=1) / (reweight_to.sum(dim=1) + 1e-12)
        #     loss = loss1 + loss2
        #     sym_loss = loss.sum()

        # adj_loss = 0
        # if len(adj_from) > 0:
        #     adj_from_th = torch.tensor(adj_from, dtype=torch.long, device=self.conf.device)
        #     obb_from = child_pred_boxes[adj_from_th, :]
        #     pc_from = transform_pc_batch(self.unit_cube, obb_from)
        #     adj_to_th = torch.tensor(adj_to, dtype=torch.long, device=self.conf.device)
        #     obb_to = child_pred_boxes[adj_to_th, :]
        #     pc_to = transform_pc_batch(self.unit_cube, obb_to)
        #     dist1, dist2 = self.chamferLoss(pc_from, pc_to)
        #     loss = (dist1.min(dim=1)[0] + dist2.min(dim=1)[0])
        #     adj_loss = loss.sum()

        # call children + aggregate losses
        # pred2allboxes = dict(); pred2allleafboxes = dict();
        # for i in range(len(matched_gt_idx)):
        #     child_losses, child_all_boxes, child_all_leaf_boxes = self.node_recon_loss(
        #         child_feats[:, matched_pred_idx[i], :], gt_data[matched_gt_idx[i]])
        #     pred2allboxes[matched_pred_idx[i]] = child_all_boxes
        #     pred2allleafboxes[matched_pred_idx[i]] = child_all_leaf_boxes
        #     all_boxes.append(child_all_boxes)
        #     all_leaf_boxes.append(child_all_leaf_boxes)
        #     box_loss = box_loss + child_losses['box']
        #     anchor_loss = anchor_loss + child_losses['anchor'] 
        # #    is_leaf_loss = is_leaf_loss + child_losses['leaf']
        #     child_exists_loss = child_exists_loss + child_losses['exists']
        #     semantic_loss = semantic_loss + child_losses['semantic']
        #     edge_exists_loss = edge_exists_loss + child_losses['edge_exists']
            # sym_loss = sym_loss + child_losses['sym']
            # adj_loss = adj_loss + child_losses['adj']

        # # for sym-edges, train subtree to be symmetric
        # for i in range(len(sym_from)):
        #     s1 = pred2allboxes[sym_from[i]].size(0)
        #     s2 = pred2allboxes[sym_to[i]].size(0)
        #     if s1 > 1 and s2 > 1:
        #         obbs_from = pred2allboxes[sym_from[i]][1:, :]
        #         obbs_to = pred2allboxes[sym_to[i]][1:, :]
        #         pc_from = transform_pc_batch(self.unit_cube, obbs_from).view(-1, 3)
        #         pc_to = transform_pc_batch(self.unit_cube, obbs_to).view(-1, 3)
        #         transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat[i][0, :, :3], 0, 1)) + \
        #                 sym_mat[i][0, :, 3].unsqueeze(dim=0).repeat(pc_from.size(0), 1)
        #         dist1, dist2 = self.chamferLoss(transformed_pc_from.view(1, -1, 3), pc_to.view(1, -1, 3))
        #         sym_loss += (dist1.mean() + dist2.mean()) * (s1 + s2) / 2

        # # for adj-edges, train leaf-nodes in subtrees to be adjacent
        # for i in range(len(adj_from)):
        #     if pred2allboxes[adj_from[i]].size(0) > pred2allleafboxes[adj_from[i]].size(0) \
        #             or pred2allboxes[adj_to[i]].size(0) > pred2allleafboxes[adj_to[i]].size(0):
        #         obbs_from = pred2allleafboxes[adj_from[i]]
        #         obbs_to = pred2allleafboxes[adj_to[i]]
        #         pc_from = transform_pc_batch(self.unit_cube, obbs_from).view(1, -1, 3)
        #         pc_to = transform_pc_batch(self.unit_cube, obbs_to).view(1, -1, 3)
        #         dist1, dist2 = self.chamferLoss(pc_from, pc_to)
        #         adj_loss += dist1.min() + dist2.min()

        # return {'box': box_loss + unused_box_loss, 'leaf': is_leaf_loss, 'anchor': anchor_loss, 
        #         'exists': child_exists_loss, 'semantic': semantic_loss,
        #         'edge_exists': edge_exists_loss, 'sym': sym_loss, 'adj': adj_loss}, \
        #                 torch.cat(all_boxes, dim=0), torch.cat(all_leaf_boxes, dim=0)
        return {'box': box_loss + unused_box_loss, 'anchor': anchor_loss, 
                 'exists': child_exists_loss, 'semantic': semantic_loss,
                 'edge_exists': edge_exists_loss}