"""
    This file defines the Hierarchy of Graph Tree class and PartNet data loader.
"""

import sys
import os
import json
import torch
import numpy as np
from torch.utils import data
from pyquaternion import Quaternion
from sklearn.decomposition import PCA
from collections import namedtuple
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from utils_sn import one_hot, tree_find
import trimesh, h5py
import kornia
import torch.nn.functional as F
import random
from scipy.spatial.transform import Rotation as R
import csv
from tqdm import tqdm

mappedfrontcate2shapenetcate ={
    "KingsizeBed": "Bed",
    "Wardrobe": "Cabinet",
    "Nightstand": "Table",
    "PendantLamp": "Lamp",
    "Bookcase": "Cabinet",
    "Desk": "Table",
    "ClassicChineseChair": "Chair",
    "CeilingLamp": "Lamp",
    "TVStand": "Table",
    "CoffeeTable": "Table",
    "CafeOfficeChair": "Chair",
    "DiningTable": "Table",
    "DiningChair": "Chair",
    "LshapedSofa": "Chair",
    "SideTable": "Table",
    "Cornercabinet": "Cabinet",
    "SideCabinet": "Cabinet",
    "Armchair": "Chair",
    "Multiseatsofa": "Chair",
    "DressingTable": "Table",
    "WineCooler": "Cabinet",
    "RoundEndTable": "Table",
    "ChildrenCabinet": "Cabinet",
    "Stool": "Chair",
    "Singlebed": "Bed",
    "KidsBed": "Bed",
    "LoveseatSofa": "Chair",
    "Jiaodeng": "Chair",
    "Yimaojia": "Cabinet",
    "DressingChair": "Chair",
    "LazySofa": "Chair",
    "Zhiwujia": "Cabinet",
    "Zhuangshijia": "Cabinet",
    "ChaiseLongueSofa": "Chair",
    "Barstool": "Chair",
    "BedFrame": "Bed",
    "Bogujia": "Cabinet",
    "BunkBed": "Bed",
    "Shelf": "Cabinet"}

cube_faces = [
    [3, 1, 0],
    [3, 2, 1],
    [5, 0, 1],
    [5, 4, 0],
    [6, 1, 2],
    [6, 5, 1],
    [7, 2, 3],
    [7, 6, 2],
    [4, 3, 0],
    [4, 7, 3],
    [7, 4, 5],
    [7, 5, 6]
]

def split_path(paths):
    filepath, tempfilename = os.path.split(paths)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension

def extract_OBB(local_transform, extents):
    corners = []
    extents = extents / 2
    corners.append([-extents[0], -extents[1], extents[2]])
    corners.append([-extents[0], extents[1], extents[2]])
    corners.append([extents[0], extents[1], extents[2]])
    corners.append([extents[0], -extents[1], extents[2]])
    corners.append([-extents[0], -extents[1], -extents[2]])
    corners.append([-extents[0], extents[1], -extents[2]])
    corners.append([extents[0], extents[1], -extents[2]])
    corners.append([extents[0], -extents[1], -extents[2]])

    corners = np.array(corners)
    four = np.ones((8, 1))
    corners = np.concatenate((corners, four), axis=1)
    # print(corners)
    # print(local_transform)
    corners = np.matmul(local_transform, np.transpose(corners))
    corners = np.transpose(corners)/np.tile(four, (1, 4))

    obb_mesh = trimesh.Trimesh(vertices = np.array(corners[:, :3]), faces=np.array(cube_faces))
    # volume = obb_mesh.volume
    # area = extents[0]*extents[2]*4
    # center = np.mean(obb_mesh.bounds, axis=0)
    # print(obb_mesh.volume)
    # print(extents[0]*extents[1]*extents[2]*8)

    return obb_mesh

def trans_extent2sn(local_transform, extent):

    obb_mesh = extract_OBB(local_transform, extent)
    # mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, _ = trimesh.sample.sample_surface(obb_mesh, 5000)
    # print(points)
    # print(np.array(points))
    pca = PCA()
    pca.fit(points)
    pcomps = pca.components_

    points_local = np.matmul(pcomps, points.transpose()).transpose()

    all_max = points_local.max(axis=0)
    all_min = points_local.min(axis=0)

    center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
    size = all_max - all_min

    xdir = pcomps[0, :]
    ydir = pcomps[1, :]

    return np.hstack([center, size, xdir, ydir]).astype(np.float32)

def twovec2quat(vecfrom, vecto):
    rotaxis = np.cross(vecfrom, vecto)
    if np.all(rotaxis == 0):
        return np.array([1.0,0,0,0])
    theta = np.arccos(1 - (np.linalg.norm(rotaxis,2)/np.linalg.norm(vecfrom,2)/np.linalg.norm(vecto,2))**2)
    rotaxis = rotaxis/np.linalg.norm(rotaxis,2)
    r = R.from_rotvec(theta * rotaxis)
    return r.as_quat()


# store a part hierarchy of graphs for a scene/room
class SceneTree(object):

    # global object category information
    part_non_leaf_sem_names = []

    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_num_sem = None

    root_sem = None
    leaf_geos_box = None
    leaf_geos_dg = None
    leaf_geos_pts = None
    cate_id = None

    @staticmethod
    def load_category_info(cat, cls):
        with open(os.path.join(BASE_DIR , '../stats/part_semanticsnew/', cat + '.txt'), 'r') as fin:
            for l in fin.readlines():
                x, y, z = l.rstrip().split()
                x = int(x)
                cls.part_name2id[y] = x
                cls.part_id2name[x] = y
                cls.part_name2cids[y] = []
                if '/' in y:
                    cls.part_name2cids['/'.join(y.split('/')[:-1])].append(x)
        #print(cls.part_name2id)
        cls.part_num_sem = len(cls.part_name2id) + 1

        for k in cls.part_name2cids:
            cls.part_name2cids[k] = np.array(cls.part_name2cids[k], dtype=np.int32)
            if len(cls.part_name2cids[k]) > 0:
                cls.part_non_leaf_sem_names.append(k)
        cls.root_sem = cls.part_id2name[1]
        print(' semantic part num (%s): %d' % (cat, (cls.part_num_sem)))

    # store a part node in the TreeA
    class Node(object):

        def __init__(self, node_id=0, is_leaf=False, is_room = True, box=None, label=None, children=None, edges=None, full_label=None, geo=None, geo_feat=None, dggeo = None, orient = None, faces = None):
            self.is_leaf = is_leaf          # store True if the part is a leaf node
            self.is_room = is_room          # store False if the node is a object
            self.node_id = node_id          # part_id in result_after_merging.json of PartNet
            self.box = box                  # box parameter for all nodes
            self.geo = geo                  # 1 x 1000 x 3 point cloud
            self.geo_id = None              # leaf node id in all node
            self.geo_box_id = None          # leaf node box id in all node
            self.geo_feat = geo_feat        # 1 x 100 geometry feature
            self.faces = faces              # facenum x 3 face index
            self.dggeo = dggeo              # 1 x pointnum x 9 deformation geo feature
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            # self.orient = orient
            self.children = [] if children is None else children
                                            # all of its children nodes; each entry is a Node instance
            self.edges = [] if edges is None else edges
                                            # all of its children relationships;
                                            # each entry is a tuple <part_a, part_b, type, params, dist>
            """
                Here defines the edges format:
                    part_a, part_b:
                        Values are the order in self.children (e.g. 0, 1, 2, 3, ...).
                        This is an directional edge for A->B.
                        If an edge is commutative, you may need to manually specify a B->A edge.
                        For example, an ADJ edge is only shown A->B,
                        there is no edge B->A in the json file.
                    type:
                        Four types considered in StructureNet: ADJ, ROT_SYM, TRANS_SYM, REF_SYM.
                    params:
                        There is no params field for ADJ edge;
                        For ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle;
                        For TRANS_SYM edge, 0-2 translation vector;
                        For REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers,
                            3-5 unit normal direction of the reflection plane.
                    dist:
                        For ADJ edge, it's the closest distance between two parts;
                        For SYM edge, it's the chamfer distance after matching part B to part A.
            """

        def get_semantic_id(self, cls):
            #return cls.part_name2id[self.full_label]
            return 0

        def get_semantic_one_hot(self, cls):
            out = np.zeros((1, cls.part_num_sem), dtype=np.float32)
            #out[0, cls.part_name2id[self.full_label]] = 1
            out[0, 0] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)

            # return torch.tensor(out, dtype=torch.float32).cuda()
            # return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)
            # return torch.tensor(out, dtype=torch.float32).to(device=self.geo.device)

        def get_box_quat1(self):
            box = self.box.cpu().numpy().squeeze()
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir /= np.linalg.norm(xdir)
            ydir = box[9:]
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)
            rotmat = np.vstack([xdir, ydir, zdir]).T
            q = Quaternion(matrix=rotmat)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            box_quat = np.hstack([center, size, quat]).astype(np.float32)
            # print(self.box.device)
            return torch.from_numpy(box_quat).view(1, -1).to(device=self.box.device)

        def get_box_quat(self):
            box = self.box.squeeze()
            # print(box)
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir = F.normalize(xdir.unsqueeze(0), p=2, dim=1)
            ydir = box[9:12]
            ydir = F.normalize(ydir.unsqueeze(0), p=2, dim=1)
            zdir = torch.cross(xdir[0], ydir[0])
            zdir = F.normalize(zdir.unsqueeze(0), p=2, dim=1)
            rotmat = torch.cat([xdir, ydir, zdir], dim = 0).transpose(1,0).unsqueeze(0).repeat(2,1,1)
            q1 = kornia.rotation_matrix_to_quaternion(rotmat, eps = 1e-6)
            quat = q1[0, [3, 0, 1, 2]]
            box_quat = torch.cat([center, size, quat])
            # self.set_from_box_quat(box_quat)
            return box_quat.view(1, -1).to(device=self.box.device)

        def set_from_box_quat1(self, box_quat):
            box_quat = box_quat.cpu().detach().numpy().squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            q = Quaternion(box_quat[6], box_quat[7], box_quat[8], box_quat[9])
            rotmat = q.rotation_matrix
            box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1).cuda()

        def set_from_box_quat(self, box_quat):
            box_quat = box_quat.squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            rotmat = kornia.quaternion_to_rotation_matrix(box_quat[[7, 8, 9, 6]])
            box = torch.cat([center, size, rotmat[:, 0].view(-1), rotmat[:, 1].view(-1)])
            self.box = box.view(1, -1).cuda()

        def to(self, device):
            if self.box is not None:
                self.box = self.box.to(device)
            for edge in self.edges:
                if 'params' in edge:
                    edge['params'].to(device)
            if self.geo is not None:
                self.geo = self.geo.to(device)
            if self.dggeo is not None:
                self.dggeo = self.dggeo.to(device)
            # if self.orient is not None:
            #     self.orient = self.orient.to(device)
            for child_node in self.children:
                child_node.to(device)

            return self

        def _to_str(self, level, pid, detailed=False):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + (' [LEAF] ' if self.is_leaf else '    ') 
            if detailed:
                out_str += 'Box('+';'.join([str(item) for item in self.box.numpy()])+')\n'
            else:
                out_str += '\n'

            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx)

            if detailed and len(self.edges) > 0:
                for edge in self.edges:
                    if 'params' in edge:
                        edge = edge.copy() # so the original parameters don't get changed
                        edge['params'] = edge['params'].cpu().numpy()
                    out_str += '  |'*(level) + '  ├' + 'Edge(' + str(edge) + ')\n'

            return out_str

        def __str__(self):
            return self._to_str(0, 0)

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        def child_adjacency(self, typed=False, max_children=None):
            if max_children is None:
                adj = torch.zeros(len(self.children), len(self.children))
            else:
                adj = torch.zeros(max_children, max_children)

            if typed:
                edge_types = ['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM']

            for edge in self.edges:
                if typed:
                    edge_type_index = edge_types.index(edge['type'])
                    adj[edge['part_a'], edge['part_b']] = edge_type_index
                    adj[edge['part_b'], edge['part_a']] = edge_type_index
                else:
                    adj[edge['part_a'], edge['part_b']] = 1
                    adj[edge['part_b'], edge['part_a']] = 1

            return adj

        def geos(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            out_geos = []; out_nodes = [];
            for node in nodes:
                if not leafs_only or node.is_leaf:
                    out_geos.append(node.geo)
                    out_nodes.append(node)
            return out_geos, out_nodes

        def boxes(self, per_node=False, leafs_only=False):
            nodes = list(reversed(self.depth_first_traversal()))
            node_boxesets = []
            boxes_stack = []
            for node in nodes:
                node_boxes = []
                for i in range(len(node.children)):
                    node_boxes = boxes_stack.pop() + node_boxes

                if node.box is not None and (not leafs_only or node.is_leaf):
                    node_boxes.append(node.box)

                if per_node:
                    node_boxesets.append(node_boxes)

                boxes_stack.append(node_boxes)

            assert len(boxes_stack) == 1

            if per_node:
                return node_boxesets, list(nodes)
            else:
                boxes = boxes_stack[0]
                return boxes

        def box_quats(self, per_node=False, leafs_only=False):
            nodes = list(reversed(self.depth_first_traversal()))
            node_boxesets = []
            boxes_stack = []
            for node in nodes:
                node_boxes = []
                for i in range(len(node.children)):
                    node_boxes = boxes_stack.pop() + node_boxes

                if node.box is not None and (not leafs_only or node.is_leaf):
                    node_boxes.append(node.get_box_quat())

                if per_node:
                    node_boxesets.append(node_boxes)

                boxes_stack.append(node_boxes)

            assert len(boxes_stack) == 1

            if per_node:
                return node_boxesets, list(nodes)
            else:
                boxes = boxes_stack[0]
                return boxes
        def graph(self, leafs_only=False):
            part_boxes = []
            part_geos = []
            edges = []
            node_ids = []
            part_sems = []

            nodes = list(reversed(self.depth_first_traversal()))

            box_index_offset = 0
            for node in nodes:
                child_count = 0
                box_idx = {}
                for i, child in enumerate(node.children):
                    if leafs_only and not child.is_leaf:
                        continue

                    part_boxes.append(child.box)
                    part_geos.append(child.geo)
                    node_ids.append(child.node_id)
                    part_sems.append(child.full_label)

                    box_idx[i] = child_count+box_index_offset
                    child_count += 1

                for edge in node.edges:
                    if leafs_only and not (
                            node.children[edge['part_a']].is_leaf and
                            node.children[edge['part_b']].is_leaf):
                        continue
                    edges.append(edge.copy())
                    edges[-1]['part_a'] = box_idx[edges[-1]['part_a']]
                    edges[-1]['part_b'] = box_idx[edges[-1]['part_b']]

                box_index_offset += child_count

            return part_boxes, part_geos, edges, node_ids, part_sems

        def edge_tensors(self, edge_types, device, type_onehot=True):
            num_edges = len(self.edges)

            # get directed edge indices in both directions as tensor
            edge_indices = torch.tensor(
                [[e['part_a'], e['part_b']] for e in self.edges] + [[e['part_b'], e['part_a']] for e in self.edges],
                device=device, dtype=torch.long).view(1, num_edges*2, 2)

            # get edge type as tensor
            edge_type = torch.tensor([edge_types.index(edge['type']) for edge in self.edges], device=device, dtype=torch.long)
            if type_onehot:
                edge_type = one_hot(inp=edge_type, label_count=len(edge_types)).transpose(0, 1).view(1, num_edges, len(edge_types)).to(dtype=torch.float32)
            else:
                edge_type = edge_type.view(1, num_edges)
            edge_type = torch.cat([edge_type, edge_type], dim=1) # add edges in other direction (symmetric adjacency)

            edge_dist_dict = dict()
            for e in self.edges:
                if 'min_dist' in e.keys():
                    edge_dist_dict[(e['part_a'], e['part_b'])] = e['min_dist']
                    edge_dist_dict[(e['part_b'], e['part_a'])] = e['min_dist']

            return edge_type, edge_indices, edge_dist_dict

        def get_subtree_edge_count(self):
            cnt = 0
            if self.children is not None:
                for cnode in self.children:
                    cnt += cnode.get_subtree_edge_count()
            if self.edges is not None:
                cnt += len(self.edges)
            return cnt

    # functions for class Tree
    def __init__(self, root):
        self.root = root

    def to(self, device):
        self.root = self.root.to(device)
        if self.leaf_geos_box is not None:
            self.leaf_geos_box = self.leaf_geos_box.to(device)
        if self.leaf_geos_dg is not None:
            self.leaf_geos_dg = self.leaf_geos_dg.to(device)
        if self.leaf_geos_pts is not None:
            self.leaf_geos_pts = self.leaf_geos_pts.to(device)
        return self

    def __str__(self):
        return str(self.root)

    def get_geo(self):
        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=self.root, parent_json=None, parent_child_idx=None)]
        # print('a')

        # traverse the tree, converting child nodes of each node to json
        geo_id = -1
        leaf_geo = []
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node
            if len(node.children) == 0 and node.geo_id is None:
                node.geo_id = 0
                leaf_geo.append(node.get_box_quat())
                break

            for child in node.children:
                # node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=None, parent_child_idx=None))
                if child.is_leaf or len(child.children) == 0:
                    geo_id = geo_id + 1
                    child.geo_id = geo_id
                    if child.geo is not None:
                        leaf_geo.append(child.geo)
                        continue

                    if child.box is not None:
                        leaf_geo.append(child.get_box_quat())
                        continue

        self.leaf_geos = torch.cat(leaf_geo, dim = 0).unsqueeze(0).to(self.root.box.device)

    def depth_first_traversal(self):
        return self.root.depth_first_traversal()

    def boxes(self, per_node=False, leafs_only=False):
        return self.root.boxes(per_node=per_node, leafs_only=leafs_only)

    def graph(self, leafs_only=False):
        return self.root.graph(leafs_only=leafs_only)

    def free(self):
        for node in self.depth_first_traversal():
            del node.geo
            del node.dggeo
            del node.geo_feat
            del node.box
            del node


# extend torch.data.Dataset class for PartNet
class FrontDatasetPartNet(data.Dataset):

    def __init__(self, root, object_list, data_features, Tree, load_geo=False):
        self.root = root
        self.data_features = data_features
        self.load_geo = load_geo
        self.Tree = Tree
        # self.current_cate_id = 0
        # self.batch_size = 32
        # self.training_cate_num = 2

        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        self.partnet_path = os.path.join(self.root, '..', 'partnetdata1')

        # accelarte the training speed, allocate all data in cpu memory
        self.all_data_obj = []
        self.object_names_new = []
        unique_id = []
        for item in tqdm(self.object_names, desc="Load Data..."):
            # print(item)
            partnet_id = item.split('--')[1]
            if partnet_id not in unique_id:
                obj = self.load_object(os.path.join(self.root, item +'.json'), \
                        Tree = self.Tree, load_geo=self.load_geo)
                self.all_data_obj.append(obj)
                self.object_names_new.append(item)
                unique_id.append(partnet_id)

        if 0 and load_geo:
            meshinfo = os.path.join(root, 'cube_meshinfo.mat')
            meshdata = h5py.File(meshinfo, mode = 'r')

            self.point_num = meshdata['neighbour'].shape[0]
            self.edge_index = np.array(meshdata['edge_index']).astype('int64')
            self.recon = np.array(meshdata['recon']).astype('float32')
            self.ref_V = np.array(meshdata['ref_V']).astype('float32')
            self.ref_F = np.array(meshdata['ref_F']).astype('int64')
            self.vdiff = np.array(meshdata['vdiff']).astype('float32')
            self.nb = np.array(meshdata['neighbour']).astype('float32')

    def __getitem__(self, index):
        if 'object' in self.data_features:
            # obj = self.load_object(os.path.join(self.root, self.object_names[index]+'.json'), \
            #         Tree = self.Tree, load_geo=self.load_geo)
            obj = self.all_data_obj[index]
        # cate_id = tree_find(self.category_names, self.object_names[index])
        # obj.cate_id = cate_id[0]
        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names_new[index],)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat

        return data_feats

    def __len__(self):
        return len(self.all_data_obj)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'), \
                Tree = self.Tree, load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_object(fn, Tree, load_geo=False):
        if load_geo:
            geo_fn = fn.replace('/hire/', '/geo_dg/').replace('.json', '.npz')
            geo_data = np.load(geo_fn, mmap_mode='r', allow_pickle=True)
        # root_path, _, _ = split_path(fn)
        # partnet_path = os.path.join(root_path, '..', 'partnetdata1')

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        leaf_geos_box = []
        leaf_geos_dg = []
        leaf_geos_pts = []
        geo_id = -1
        geo_box_id = -1
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            # is_room = node_json['type'] in list(Tree.room_name2id.keys()) + list(Tree.region_name2id.keys()) + ['house']
            # is_part = node_json['full_type'].split('--')[-1] in list(Tree.part_name2id.keys())
            # load the partnet hire json
            # if load_geo and not is_room and not is_part and 'future_id' in node_json.keys() and node_json['future_id'] != '':
            #     # print(node_json['type'])
            #     partnetid = futureid2partnetid[node_json['future_id']]
            #     # partnet_json = json.load(open(os.path.join(partnet_path, 'hire', partnetid + '.json'), 'r'))
            #     geo_data_object = np.load(os.path.join(partnet_path, 'geo_pc', partnetid + '.npz'), mmap_mode='r', allow_pickle=True)
                # node_json['id'] = partnet_json['id']
                # node_json['type'] = partnet_json['type']
                # node_json['child_region'] = partnet_json['child_region']
                # node_json['box'] = partnet_json['box']
                # if 'edges' in partnet_json.keys():
                #     node_json['edges'] = partnet_json['edges']
                # else:
                #     node_json['edges'] = []

            # if node_json['full_type'].split('--')[-1] in front2shapecatemap.keys():
            #     node_label = front2shapecatemap[node_json['full_type'].split('--')[-1]]
            # else:
            # node_label = node_json['full_type'].split('--')[-1]
            # if len(node_json['full_type'].split('--')) == 4 and node_label in mappedfrontcate2shapenetcate.keys():
            #     node_label = mappedfrontcate2shapenetcate[node_label]

            node = Tree.Node(
                node_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])

            # print(node_json['id'])
            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)
            if load_geo:
                # print(is_room)
                node.geo = torch.tensor(geo_data['partsV'][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                LOGR = torch.tensor(geo_data['LOGR'][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                S = torch.tensor(geo_data['S'][node_json['id']], dtype=torch.float32).view(1, -1, 6)
                node.dggeo = torch.cat((LOGR, S), 2)
                node.faces = torch.tensor(geo_data['F'], dtype=torch.int32)

            # if load_geo and node_label not in Tree.part_non_leaf_sem_names and not is_part:
                # if 'child_region' not in node_json or len(node_json['child_region']) == 0:
                # node.geo = torch.tensor(geo_data_scene['pcs1024'][()][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                # LOGR = torch.tensor(geo_data['LOGR'][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                # S = torch.tensor(geo_data['S'][node_json['id']], dtype=torch.float32).view(1, -1, 6)
                # node.dggeo = torch.cat((LOGR, S), 2)
                # node.faces = torch.tensor(geo_data['F'], dtype=torch.int32)
                # node.geo = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1, 3)

            if 'box' in node_json:
                # if not is_part and len(node_json['box']) == 19:
                #     box = np.array(node_json['box'])
                #     node.box = torch.from_numpy(trans_extent2sn(box[:-3].reshape(-1, 4), box[-3:])).to(dtype=torch.float32)
                #         # node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)
                # else:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)
                geo_box_id = geo_box_id + 1
                node.geo_box_id = geo_box_id

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            # if 'model_orient' in node_json:
            #     node.orient = torch.tensor(twovec2quat(np.array([0.0,0.0,1.0]), np.array(node_json['model_orient']))).to(dtype=torch.float32).view(1, -1)

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            # if ('child_region' not in node_json):
            #     # get the leaf node id in all node list
            #     if load_geo:
            #         leaf_geos_dg.append(torch.from_numpy(np.expand_dims(node.dggeo, axis = 0)).to(dtype=torch.float32))
            #         leaf_geos_pts.append(torch.from_numpy(np.expand_dims(node.geo, axis = 0)).to(dtype=torch.float32))
            #         leaf_geos_box.append(node.get_box_quat())
            #         geo_id = geo_id + 1
            #         node.geo_id = geo_id
            #         geo_box_id = geo_box_id + 1
            #         node.geo_box_id = geo_box_id
            #     else:
            #         leaf_geos_box.append(node.get_box_quat())
            #         geo_box_id = geo_box_id + 1
            #         node.geo_box_id = geo_box_id

            if parent is None:
                root = node
                # root.full_label = root.label

                # for five cates
                # root.full_label = 'object/' + root.label
                root.full_label = root.label
                # root.lable = root.full_label.split('--')[-1]
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                # if not is_room:

                # if is_part:
                #     node.full_label = curobj_label + '--' + node_json['full_type']
                # else:
                #     node.full_label = parent.full_label + '--' + node.label
                #     curobj_label = node.full_label
                # node.label = node.full_label.split('--')[-1]
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)
        # obj.leaf_geos_box = torch.cat(leaf_geos_box, dim = 0).unsqueeze(0)
        # obj.leaf_geos_dg = torch.cat(leaf_geos_dg, dim = 1)
        # obj.leaf_geos_pts = torch.cat(leaf_geos_pts, dim = 1)
        # print(obj.leaf_geos)

        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.node_id,
                # 'is_room' : node.is_room,
                'type': f'{node.label if node.label is not None else ""}',
                'label' : node.full_label}

            if node.geo is not None:
                node_json['geo'] = node.geo.detach().cpu().numpy().reshape(-1).tolist()

            if node.box is not None:
                node_json['box'] = node.box.detach().cpu().numpy().reshape(-1).tolist()

            # if node.orient is not None:
            #     if 'model_orient' in node_json.keys():
            #         node_json['model_orient'] = node.orient.cpu().numpy().reshape(-1).tolist()
            #     else:
            #         node_json['model_orient'] = np.array([0.0,0.0,1.0]).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if len(node.edges) > 0:
                node_json['edges'] = []
            for edge in node.edges:
                node_json['edges'].append(edge)
                if 'params' in edge:
                    node_json['edges'][-1]['params'] = node_json['edges'][-1]['params'].cpu().numpy().reshape(-1).tolist()

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        # obj_json['type'] = 'house'
        with open(fn, 'w') as f:
            json.dump(obj_json, f)