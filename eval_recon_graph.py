"""
    This is the main tester script for box-shape reconstruction evaluation.
    Use scripts/eval_recon_box_ae_chair.sh to run.
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from config import add_eval_args
from chamfer_distance import ChamferDistance
import scene_graph_dataset
import scene_graph_structurenet
#import compute_sym

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

chamferLoss = ChamferDistance()

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
if os.path.exists(os.path.join(conf.result_path, conf.exp_name)):
    response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
    if response != 'y':
        sys.exit()
    shutil.rmtree(os.path.join(conf.result_path, conf.exp_name))

# create a new directory to store eval results
os.makedirs(os.path.join(conf.result_path, conf.exp_name))

# create models
encoder = models.GNNEncoderStructureNet(conf.feature_size, conf.hidden_size, conf.node_symmetric_type, conf.edge_symmetric_type, conf.num_gnn_iterations, 3)
decoder = models.GNNDecoderStructureNet(conf.feature_size, conf.hidden_size, conf.max_child_num,conf.edge_symmetric_type, conf.num_dec_gnn_iterations, 3)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

# load pretrained model
__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

# create dataset and data loader
#data_features = ['object', 'name']
dataset = scene_graph_dataset.SceneGraphDataSet(conf.data_path, 'MasterBedroom',valid=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_feats)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# load unit cube pc
unit_cube = torch.from_numpy(utils.load_pts('cube.pts')).to(device)

def boxLoss(box_feature, gt_box_feature):
    pred_box_pc = utils.transform_pc_batch(unit_cube, box_feature)
    pred_reweight = utils.get_surface_reweighting_batch(box_feature[:, 3:6], unit_cube.size(0))
    gt_box_pc = utils.transform_pc_batch(unit_cube, gt_box_feature)
    gt_reweight = utils.get_surface_reweighting_batch(gt_box_feature[:, 3:6], unit_cube.size(0))
    dist1, dist2 = chamferLoss(gt_box_pc, pred_box_pc)
    loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
    loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
    loss = (loss1 + loss2) / 2
    return loss

# def compute_binary_diff(pred_node):
#     if pred_node.is_leaf:
#         return 0, 0
#     else:
#         binary_diff = 0; binary_tot = 0;

#         # all children
#         for cnode in pred_node.children:
#             cur_binary_diff, cur_binary_tot = compute_binary_diff(cnode)
#             binary_diff += cur_binary_diff
#             binary_tot += cur_binary_tot

#         # current node
#         if pred_node.edges is not None:
#             for edge in pred_node.edges:
#                 pred_part_a_id = edge['part_a']
#                 obb1 = pred_node.children[pred_part_a_id].box.cpu().numpy()
#                 obb_quat1 = pred_node.children[pred_part_a_id].get_box_quat().cpu().numpy()
#                 mesh_v1, mesh_f1 = utils.gen_obb_mesh(obb1)
#                 pc1 = utils.sample_pc(mesh_v1, mesh_f1, n_points=500)
#                 pc1 = torch.tensor(pc1, dtype=torch.float32, device=device)
#                 pred_part_b_id = edge['part_b']
#                 obb2 = pred_node.children[pred_part_b_id].box.cpu().numpy()
#                 obb_quat2 = pred_node.children[pred_part_b_id].get_box_quat().cpu().numpy()
#                 mesh_v2, mesh_f2 = utils.gen_obb_mesh(obb2)
#                 pc2 = utils.sample_pc(mesh_v2, mesh_f2, n_points=500)
#                 pc2 = torch.tensor(pc2, dtype=torch.float32, device=device)
#                 if edge['type'] == 'ADJ':
#                     dist1, dist2 = chamferLoss(pc1.view(1, -1, 3), pc2.view(1, -1, 3))
#                     binary_diff += (dist1.sqrt().min().item() + dist2.sqrt().min().item()) / 2
#                 elif 'SYM' in edge['type']:
#                     if edge['type'] == 'TRANS_SYM':
#                         mat1to2, _ = compute_sym.compute_trans_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
#                     elif edge['type'] == 'REF_SYM':
#                         mat1to2, _ = compute_sym.compute_ref_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
#                     elif edge['type'] == 'ROT_SYM':
#                         mat1to2, _ = compute_sym.compute_rot_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
#                     else:
#                         assert 'ERROR: unknown symmetry type: %s' % edge['type']
#                     mat1to2 = torch.tensor(mat1to2, dtype=torch.float32, device=device)
#                     transformed_pc1 = pc1.matmul(torch.transpose(mat1to2[:, :3], 0, 1)) + \
#                             mat1to2[:, 3].unsqueeze(dim=0).repeat(pc1.size(0), 1)
#                     dist1, dist2 = chamferLoss(transformed_pc1.view(1, -1, 3), pc2.view(1, -1, 3))
#                     loss = (dist1.sqrt().mean() + dist2.sqrt().mean()) / 2
#                     binary_diff += loss.item()
#                 else:
#                     assert 'ERROR: unknown symmetry type: %s' % edge['type']
#                 binary_tot += 1

#         return binary_diff, binary_tot

# test over all test shapes
num_batch = len(dataloader)
chamfer_dists = []
structure_dists = []
edge_precisions = []
edge_recalls = []
pred_binary_diffs = []
gt_binary_diffs = []
with torch.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        data = batch[0][0]
        edge_index = batch[1][0]
        edge_type = batch[2][0]
        data_input = np.zeros((20,np.size(data,1)))
        data_input[:np.size(data,0),:] = data

        data_exist = np.zeros((20,1))
        data_exist[0:np.size(data,0),:] = np.ones((np.size(data,0),1))

        data_loss_input = torch.tensor(data.reshape(1,-1,14 +128 +8).astype("float32"),device=device)
        data_input = torch.tensor(data_input.astype("float32"),device=device)
        data_exist = torch.tensor(data_exist.astype("float32"),device=device)
        edge_index = torch.tensor(edge_index.astype("long"),device=device)
        edge_type = torch.tensor(edge_type.astype("long"),device=device)

        #print(edge_index.T.long().reshape(1,-1,2))

        root_code = encoder.forward(data_input.reshape(1,20,-1),data_exist.reshape(1,20,1),edge_index.T.long().reshape(1,-1,2),edge_type.T.reshape(1,-1,3))

        child_feats, child_sem_logits, child_exists_logits, edge_exists_logits= decoder.forward(parent_feature=root_code)
        obj_losses = decoder.graph_recon_loss(node_latent=root_code, gt_data=data_loss_input, gt_edge_index = edge_index.T.long().reshape(1,-1,2),gt_edge_type=edge_type.reshape(1,-1,3))
        print('[%d/%d] ' % (batch_ind, num_batch))

        pred_boxes = decoder.box_decoder(child_feats.view(-1,128))
        root_code_pred = decoder.mlp_rootcode(child_feats.view(-1,128))
        moveable_pred = decoder.moveabledecoder(child_feats.view(-1, 128))
        root_code_pred = root_code_pred.view(-1,128)
        moveable_pred = moveable_pred.view(-1,8)

        data_pred = []
        edge_index_pred = []

        exist_childs = []
        for ci in range(child_feats.shape[1]):
            if torch.sigmoid(child_exists_logits[:, ci, :]).item() > 0.5:
                exist_childs.append(ci)
                feat_tmp = pred_boxes[ci,:].cpu().numpy()
                rootcode_tmp = root_code_pred[ci,:].cpu().numpy()
                moveable_tmp = moveable_pred[ci,:].cpu().numpy()
                sem = np.zeros(7)
                idx = np.argmax(child_sem_logits[0,ci,:].cpu().numpy())
                sem[idx] = 1
                data_pred.append(np.concatenate([feat_tmp,sem,rootcode_tmp,moveable_tmp]))

        for cx in range(child_feats.shape[1]):
            for cy in range(child_feats.shape[1]):
                if torch.sigmoid(torch.max(edge_exists_logits[0, cx,cy, :])).item() > 0.5:
                    try:
                        edge_index_pred.append(np.array([exist_childs.index(cx),exist_childs.index(cy)]))
                    except:
                        continue
        #print(data_pred)
        #print(edge_index_pred)
        print(obj_losses)
        edge_index_pred = np.array(edge_index_pred).T

        pred_graph = scene_graph_structurenet.BasicSceneGraph(np.array(data_pred),edge_index_pred,None,'none')
        pred_graph.visualize()
        pred_graph.visualize2D()

        gt_graph = scene_graph_structurenet.BasicSceneGraph(data,edge_index.cpu().numpy(),None,'none')
        gt_graph.visualize()
        gt_graph.visualize2D()
        # print(child_feats)
        # print(child_sem_logits.shape)
        # print(child_exists_logits.shape)
        # print(edge_exists_logits.shape)

        
     