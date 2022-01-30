import json
import os
import numpy as np
import torch
from common import *
import trimesh
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx
from scipy.spatial.transform import Rotation as R

import pickle
from utils import load_pts, transform_pc1
import chamfer_distance
import torch
from structurenet.code.model_structure_net_box import RecursiveEncoder,RecursiveDecoder
from structurenet.code.vis_utils import draw_partnet_objects
from structurenet.code.data import SceneTree,FrontDatasetPartNet
import random
from treebase import Chair, Table, Bed, Lamp, Storage_Furniture
from collections import namedtuple
import math
import partnetmobildataset_utils
import traceback
from shapely.geometry.polygon import Polygon
from descartes.patch import PolygonPatch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config():
  def __setitem__(self,k,v):
    self.__setattr__(k,v)
  def __getitem__(self,k):
    try:
      return self.__getattribute__(k)
    except AttributeError:
      return None
unit_cube = load_pts('cube.pts')

encoders = {}
decoders = {}

Object_cates = ['Chair', 'Table', 'Bed','Lamp' ,'Storage_Furniture']
cates_class = [Chair, Table, Bed,Lamp, Storage_Furniture]


PartTrees = {}
for id, cate_id in enumerate(Object_cates):
    cur_Tree = cates_class[id]
    cur_Tree.load_category_info(cate_id, cur_Tree)


# for cat in category_class:
#     if cat.find('Moveable') != -1 : continue

#     config = torch.load(BASE_DIR + '/structurenet/data/models/PartNetBox_vae_' + cat + '/conf.pth')
#     #print(config)
#     #print(cat)
#     encoders[cat] = RecursiveEncoder(config,variational=True,Tree=cates_class[Object_cates.index(cat)])
#     encoders[cat].load_state_dict(torch.load(BASE_DIR + '/structurenet/data/models/PartNetBox_vae_' + cat + '/net_encoder.pth'), strict=True)
#     decoders[cat] = RecursiveDecoder(config,Tree=cates_class[Object_cates.index(cat)])
#     decoders[cat].load_state_dict(torch.load(BASE_DIR + '/structurenet/data/models/PartNetBox_vae_' + cat + '/net_decoder.pth'), strict=True)


# def load_object(fn, load_geo=False):
#     if load_geo:
#         geo_fn = fn.replace('_hier', '_geo').replace('json', 'npz')
#         geo_data = np.load(geo_fn)

#     with open(fn, 'r') as f:
#         root_json = json.load(f)

#     # create a virtual parent node of the root node and add it to the stack
#     StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
#     stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]
#     Tree_tmp = Tree(root=None)
#     for cat in category_class:
#         if fn.find(cat.lower()) != -1: break
#     Tree_tmp.load_category_info(cat=cat)
#     root = None
#     # traverse the tree, converting each node json to a Node instance
#     while len(stack) > 0:
#         stack_elm = stack.pop()

#         parent = stack_elm.parent
#         parent_child_idx = stack_elm.parent_child_idx
#         node_json = stack_elm.node_json

#         node = Tree.Node(
#             part_id=node_json['id'],
#             is_leaf=('children' not in node_json),
#             label=node_json['label'],tree=Tree_tmp)

#         if 'geo' in node_json.keys():
#             node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)

#         if load_geo:
#             node.geo = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1, 3)

#         if 'box' in node_json:
#             node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)

#         if 'children' in node_json:
#             for ci, child in enumerate(node_json['children']):
#                 stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

#         if 'edges' in node_json:
#             for edge in node_json['edges']:
#                 if 'params' in edge:
#                     edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
#                 node.edges.append(edge)

#         if parent is None:
#             root = node
#             root.full_label = root.label
#         else:
#             if len(parent.children) <= parent_child_idx:
#                 parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
#             parent.children[parent_child_idx] = node
#             node.full_label = parent.full_label + '/' + node.label

#     obj = Tree(root=root)

#     return obj

def min_dist(data_1,data_2):
    pc_1 = transform_pc1(torch.tensor(unit_cube), torch.tensor(data_1)).float()
    pc_2 = transform_pc1(torch.tensor(unit_cube), torch.tensor(data_2)).float()
    dist1, dist2 = chamfer_distance.ChamferDistance().forward(pc_1.reshape(1,-1,3), pc_2.reshape(1,-1,3))
    return torch.min(dist1)

def make_dense_edge_index(node_num: int):
    ret = np.zeros((2,node_num * (node_num - 1)))
    for i in range(node_num):
        idx = 0
        for j in range(node_num):
            if j == i:
                continue
            ret[0, i * (node_num - 1 ) + idx] = i
            ret[1, i * (node_num - 1) + idx] = j
            idx += 1
    return ret

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

class BasicSceneGraph:
    def __init__(self, data: np.array,subbox_data:np.array,subbox_length:list, edge_index: np.array,edge_type: np.array,room_type:str, wall:tuple,wall_countour, room_id:str):
        self._data = data
        self._edge_index = edge_index
        self._room_type = room_type
        self._edge_type = edge_type
        self._subbox_data = subbox_data
        self._subbox_length = subbox_length
        self.wall = wall
        self.wall_countour = wall_countour
        self.room_id = room_id


    def visualize2D(self,save_path = None):
        fig = plt.figure("2D Top-Down")
        fig.clear()
        ax = fig.add_subplot(111, aspect='equal')
        G = networkx.DiGraph()
         # wall_vert_center = (np.max(wall_verts,axis=0) + np.min(wall_verts,axis=0)) / 2
       # wall_verts = (wall_verts - wall_vert_center) * 1.05 + wall_vert_center
        wall_countour = self.wall_countour
        wall_vert_center = (np.max(wall_countour,axis=0) + np.min(wall_countour,axis=0)) / 2
        wall_countour = (wall_countour - wall_vert_center) * 1.05 + wall_vert_center
        ax.scatter(wall_countour[:,0],wall_countour[:,2],linewidths=0.1,marker='.')
        centers = []
        for object_index in range(np.size(self._data,0)):
            object_descriptor = self._data[object_index]
            center = object_descriptor[0:3]
            print(center)
            extents = object_descriptor[3:6]

            translation = trimesh.transformations.translation_matrix(center)
            rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            minv = np.min(b.vertices,axis=0)
            maxv = np.max(b.vertices,axis=0)
    
            ax.add_line(Line2D([minv[0],maxv[0]],[minv[2],minv[2]]))
            ax.add_line(Line2D([maxv[0],maxv[0]],[minv[2],maxv[2]]))
            ax.add_line(Line2D([maxv[0],minv[0]],[maxv[2],maxv[2]]))
            ax.add_line(Line2D([minv[0],minv[0]],[maxv[2],minv[2]]))

            centerv = (minv + maxv) / 2
            centers.append(centerv)

            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]

            plt.text(minv[0],maxv[2],category_str)

            G.add_node(object_index,label = category_str)
        if self._edge_index.size != 0:
            for edge_index in range(np.size(self._edge_index,1)):
                G.add_edge(self._edge_index[0,edge_index], self._edge_index[1,edge_index])
                center_orig = centers[self._edge_index[0,edge_index]]
                center_targ = centers[self._edge_index[1,edge_index]]
                ax.add_line(Line2D([center_orig[0],center_targ[0]],[center_orig[2],center_targ[2]]))

        plt.autoscale()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            
            plt.show()     
            pos = networkx.spectral_layout(G)
            node_labels = networkx.get_node_attributes(G, 'label')
            networkx.draw_networkx_labels(G, pos, labels=node_labels)
            networkx.draw(G)
            plt.show()

        
    def visualize(self, save_path = None):
        print(self.room_id)
        scene = trimesh.Scene()
       # wall_verts = self.wall[0]
       # wall_vert_center = (np.max(wall_verts,axis=0) + np.min(wall_verts,axis=0)) / 2
       # wall_verts = (wall_verts - wall_vert_center) * 1.05 + wall_vert_center
        wall_countour = self.wall_countour
        wall_vert_center = (np.max(wall_countour,axis=0) + np.min(wall_countour,axis=0)) / 2
        wall_countour = (wall_countour - wall_vert_center) * 1.05 + wall_vert_center
      #  wall_mesh = trimesh.Trimesh(vertices=wall_verts,faces=self.wall[1])
        print('xxxxxxxxxx')
       # print(wall_mesh.bounding_box.primitive.transform)
      #  scene.add_geometry(wall_mesh)
        G = networkx.DiGraph()
        subbox_length_sum = [0]
        sum = 0
        for length in self._subbox_length:
            sum+= length
            subbox_length_sum.append(sum)
        for object_index in range(np.size(self._data,0)):
            #if object_index != 3 and object_index != 4:continue
            object_descriptor = self._data[object_index]
            
            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]
            G.add_node(object_index,label = category_str)


            #root_code = torch.tensor(object_descriptor[14:142]).float().reshape(1,128)
            #print(root_code.shape)
            if category_str.find('Moveable') == -1:
                #obj = decoders[category_str].decode_structure(root_code,max_depth=100)
                boxes = self._subbox_data[subbox_length_sum[object_index]:subbox_length_sum[object_index+1]]
                scene_tmp = trimesh.Scene()
                for box in boxes:
                    center = box[0:3]
                    extents = box[3:6]
                    rot = box[6:10]

                    translation = trimesh.transformations.translation_matrix(center)
                    rotation = trimesh.transformations.quaternion_matrix(rot)
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
                    b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                    scene_tmp.add_geometry(b)
                    #scene_tmp.rezero()
                bounds = (scene_tmp.bounds)
                tmp_extents = bounds[1,:] - bounds[0,:]
                tmp_center = (bounds[1,:] + bounds[0,:] ) / 2
                scene_tmp.apply_transform(trimesh.transformations.translation_matrix(-tmp_center))
                #print(object_descriptor[3:6] / tmp_extents)
                scene_tmp.apply_transform(trimesh.transformations.compose_matrix(scale=(object_descriptor[3:6]) / (tmp_extents)))
                #print(object_descriptor[6])
                scene_tmp.apply_transform(rotation_matrix(np.array([0,1,0]),object_descriptor[6]))
                #print(object_descriptor[6:10])
                scene_tmp.apply_transform(trimesh.transformations.translation_matrix(object_descriptor[0:3]))
                #print(trimesh.transformations.euler_from_quaternion(object_descriptor[6:10]))
                scene.add_geometry(scene_tmp)
            else:
                #print(1)
                center = object_descriptor[0:3]
                extents = object_descriptor[3:6]    
                #print(object_descriptor)
                #print(object_descriptor[14:22])
                if category_str == 'Moveable_Slider':
                    translation = trimesh.transformations.translation_matrix(center)
                    rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])

                    slide_direction = object_descriptor[14:17]

                    #slide_direction = np.dot(slide_direction,rotation[0:3,0:3])
                    slide_direction = slide_direction / np.linalg.norm(slide_direction)
                    translation_1 = trimesh.transformations.translation_matrix(slide_direction * object_descriptor[17])
                    translation_2 = trimesh.transformations.translation_matrix(slide_direction * object_descriptor[18])
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,translation_1)))
                    b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                    scene.add_geometry(b)

                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,translation_2)))
                    b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                    scene.add_geometry(b)

                else:
                    translation = trimesh.transformations.translation_matrix(center)
                    rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])

                    rotation_orig = object_descriptor[14:17]

                    rotation_direction = object_descriptor[17:20]
                    print(rotation_direction)
                   # rotation_orig = np.dot(rotation_orig,rotation[0:3,0:3])
                    #rotation_orig = rotation_orig / np.linalg.norm(rotation_orig)
                    
                    #print(rotation_orig)
                   # print(rotation_direction)
                    print(object_descriptor)
                    rotation_1 =  trimesh.transformations.rotation_matrix(object_descriptor[20] , rotation_direction,rotation_orig)

                    rotation_2 =  trimesh.transformations.rotation_matrix(object_descriptor[21] , rotation_direction,rotation_orig)

                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,rotation_1)))
                    b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                    scene.add_geometry(b)

                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,rotation_2)))
                    b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                    scene.add_geometry(b)



        if self._edge_index.size != 0:
            for edge_index in range(np.size(self._edge_index,1)):
                G.add_edge(self._edge_index[0,edge_index], self._edge_index[1,edge_index])

        if save_path is not None:
            scene.set_camera(distance = 8)
            image_data = scene.save_image(resolution = (512,512))
            with open(save_path,'wb') as f:
                f.write(image_data)
        else:
            print('yyyyyyyyyyyyyyyy')
            print(scene.bounds)
            scene.show()

            pos = networkx.random_layout(G)
            node_labels = networkx.get_node_attributes(G, 'label')
            networkx.draw_networkx_labels(G, pos, labels=node_labels)
            networkx.draw(G)
            plt.show()
    
    #output a scene configuration for moveit
    def OutputMoveitSceneFile(self,filename):
        with open(filename,'w') as f:

            for object_index in range(np.size(self._data,0)):
                object_descriptor = self._data[object_index]
                category_onehot = list(object_descriptor[7:])
                category_str = category_class[category_onehot.index(1)]
                if(category_str != 'Moveable'):
                    
                    category_onehot = list(object_descriptor[7:])
                    category_str = category_class[category_onehot.index(1)]
                    


                    root_code = torch.tensor(object_descriptor[13:]).float().reshape(1,128)
                    #print(root_code.shape)
                    obj = decoders[category_str].decode_structure(root_code,max_depth=100)
                    boxes = (obj.root.box_quats(leafs_only=True))
                    scene_tmp = trimesh.Scene()
                    for box in boxes:
                        box = box.cpu().detach().numpy().flatten()
                        center = box[0:3]
                        extents = box[3:6]
                        rot = box[6:10]
                        
                        translation = trimesh.transformations.translation_matrix(center)
                        rotation = trimesh.transformations.quaternion_matrix(rot)
                        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
                        b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                        scene_tmp.add_geometry(b)
                    #scene_tmp.rezero()
                    bounds = (scene_tmp.bounds)
                    tmp_extents = bounds[1,:] - bounds[0,:]
                    tmp_center = (bounds[1,:] + bounds[0,:] ) / 2


                    for box in boxes:
                        box = box.cpu().detach().numpy().flatten()
                        center = box[0:3]
                        extents = box[3:6]
                        rot = box[6:10]
                        center = center - tmp_center
                        center = center * (object_descriptor[3:6]) / (tmp_extents)
                        extents = extents * (object_descriptor[3:6]) / (tmp_extents)
                        center = np.dot(rotation_matrix(np.array([0,1,0]),object_descriptor[6])[0:3,0:3],center)
                        center = center + object_descriptor[0:3]
                        center = center[[0,2,1]]
                        matrix = trimesh.transformations.quaternion_matrix(rot)
                        matrix = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675),np.dot(rotation_matrix(np.array([0,1,0]),object_descriptor[6]),matrix))
                        rot = trimesh.transformations.quaternion_from_matrix(matrix)
                        f.write('box ' + str(center[0]) + ' ' +  str(center[1]) + ' ' +str(center[2]) + ' ' + str(extents[0]) + ' ' + str(extents[1]) + ' ' +str(extents[2]) + ' ' + str(rot[0]) + ' ' +str(rot[1]) + ' ' +str(rot[2]) +' ' + str(rot[3]) + '\n')
                else:
                    center = object_descriptor[0:3]
                    extents = object_descriptor[3:6]    
                    if object_descriptor[13] == 0:

                        slide_direction = object_descriptor[14:17]
                        #slide_direction = np.dot(slide_direction,rotation[0:3,0:3])
                        slide_direction = slide_direction / np.linalg.norm(slide_direction)
                        translation_1 = trimesh.transformations.translation_matrix(slide_direction * object_descriptor[17])
                        translation_2 = trimesh.transformations.translation_matrix(slide_direction * object_descriptor[18])

                        center_orig = center + slide_direction * object_descriptor[17]

                        matrix = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675),(rotation_matrix(np.array([0,1,0]),object_descriptor[6])))
                        rot = trimesh.transformations.quaternion_from_matrix(matrix)

                        f.write('moveable ' + str(center[0]) + ' ' +  str(center[2]) + ' ' +str(center[1]) + ' ' + str(extents[0]) + ' ' + str(extents[1]) + ' ' +str(extents[2]) + ' ' + str(rot[0]) + ' ' +str(rot[1]) + ' ' +str(rot[2]) +' ' + str(rot[3]) + '\n')

                        # get the slide line origin in local frame
                        slide_orig = slide_direction * (extents - object_descriptor[17])
                        slide_dest = slide_orig + slide_direction * object_descriptor[18]
                        slide_orig = np.dot(matrix[:3,:3], slide_orig) + center[[0,2,1]]
                        slide_dest = np.dot(matrix[:3,:3], slide_dest) + center[[0,2,1]]

                        f.write('path line ' + str(slide_orig[0]) + ' ' +  str(slide_orig[1]) + ' ' +str(slide_orig[2]) + ' ' + str(slide_dest[0]) + ' ' + str(slide_dest[1]) + ' ' +str(slide_dest[2]) + '\n')
                        
                    else:
                        rotation_orig = object_descriptor[14:17]
                        rotation_direction = object_descriptor[17:20]
                        # rotation_orig = np.dot(rotation_orig,rotation[0:3,0:3])
                        #rotation_orig = rotation_orig / np.linalg.norm(rotation_orig)
                        
                        rotation_1 =  trimesh.transformations.rotation_matrix(object_descriptor[20] / 180 * math.pi, rotation_direction,rotation_orig)

                        matrix = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675),(rotation_matrix(np.array([0,1,0]),object_descriptor[6])))
                        rot = trimesh.transformations.quaternion_from_matrix(matrix)

                        f.write('moveable ' + str(center[0]) + ' ' +  str(center[2]) + ' ' +str(center[1]) + ' ' + str(extents[0]) + ' ' + str(extents[1]) + ' ' +str(extents[2]) + ' ' + str(rot[0]) + ' ' +str(rot[1]) + ' ' +str(rot[2]) +' ' + str(rot[3]) + '\n')

                        rotdir_main_idx = np.argmax(np.abs(rotation_direction ))
                        extent_min_idx = np.argmin(extents)
                        extents_tmp = extents
                        rotation_orig[[rotdir_main_idx,extent_min_idx]] = [0,0]
                        radius = 2* np.linalg.norm(rotation_orig)
                        rotation_center = rotation_orig
                        rotation_orig = -rotation_orig

                        rotation_direction = np.dot(matrix[:3,:3], rotation_direction)
                        rotation_center = np.dot(matrix[:3,:3], rotation_center) + center[[0,2,1]]
                        rotation_orig = np.dot(matrix[:3,:3], rotation_orig) + center[[0,2,1]]

                       

                        f.write('path arc ' + str(rotation_center[0]) + ' ' +  str(rotation_center[1]) + ' ' + str(rotation_center[2]) + ' ' + str(rotation_orig[0]) + ' ' +  str(rotation_orig[1]) + ' ' +str(rotation_orig[2]) + ' ' + str(radius) + ' ' +  str(object_descriptor[21] - object_descriptor[20]) + ' ' + str(rotation_direction[0]) + ' ' +str(rotation_direction[1]) + ' ' +str(rotation_direction[2]) +' '  '\n')
                
    @staticmethod
    def getGraphsfromFRONTFile(path:str):
        # cope with 3D boundingbox now
        moveable_list = os.listdir('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/')
        with open('/mnt/2/sjm_env/SceneEvolutionGraph/replacement_new_rl.json','r') as f: 
            mesh_matching = json.load(f)
        with open(path) as f:
            json_data = json.load(f)
        furniture_list = json_data['furniture']
        room_list = json_data['scene']['room']
        
        with open(FUTURE_PATH + '/model_info.json') as f:
            model_category_data_raw = json.load(f)
        model_category_data = {}
        for model in model_category_data_raw:
            model_category_data[model['model_id']] = model['category']
        
        ret = []
        scenename = json_data['uid']
        print(scenename)

        # 3D-FUTURE: Z+
        # partnet:Z+
        for room in room_list:
            print("--------------------------------------------")
            print(room['instanceid'].replace('-','--'))
            data = []
            normalize_scene = trimesh.Scene()
            furniture_category_list = []
            furniture_root_code_list = []
            angle_list = []
            choiced_list = {}
            subbox_data = []
            subbox_lengths = []

            obj_partnet_list = []
            obj_orig_extents_list = []
            roomtype = room['instanceid'].replace('-','--')
            room_id = scenename + '--' + roomtype
            if not os.path.exists('/home/yangjie/208/yangjie/scene/data/3D-FRONTnew/floor_data/mesh/'+ scenename + '--' + roomtype + '.obj'):
                print('/home/yangjie/208/yangjie/scene/data/3D-FRONTnew/floor_data/mesh/'+ scenename + '--' + roomtype + '.obj')
                continue
            
            wall_countour_obj = trimesh.load('/home/yangjie/208/yangjie/scene/data/3D-FRONTnew/floor_data/mesh/'+ scenename + '--' + roomtype + '.obj',process=False)
            wall_countour = wall_countour_obj.vertices[:1024]

            wall_vert_center = (np.max(wall_countour,axis=0) + np.min(wall_countour,axis=0)) / 2
           

            wall_countour2 = (wall_countour - wall_vert_center) * 1.05 + wall_vert_center
            wall_countour_outer = (wall_countour2 - wall_vert_center) * 6 + wall_vert_center
            wall_countour_outer = np.concatenate([wall_countour_outer,wall_countour_outer[0].reshape(1,-1)],axis=0)
            wall_not_valid =False
            try:
                poly = Polygon(wall_countour_outer[:,[0,2]],[np.concatenate([wall_countour2,wall_countour2[0].reshape(1,-1)],axis=0)[:,[0,2]]])
                wall_collision_mesh = trimesh.creation.extrude_polygon(poly,20)
                trans_toorig =  trimesh.transformations.translation_matrix([0,4,0])
                rot_yz = rotation_matrix(np.array([1,0,0]),3.1415926535 / 2)
                wall_collision_mesh = wall_collision_mesh.apply_transform(np.dot(trans_toorig,rot_yz))
                wall_ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(wall_collision_mesh)

            except:
                # p1 = trimesh.load_path(wall_countour)
                # p2 = trimesh.load_path(wall_countour_outer)
                # p1.show()
                # p2.show()
                wall_not_valid =True
        # patch = PolygonPatch(poly)
        #  ax.add_patch(patch)

        # plt.show()

            for object_in_room in room['children']:
                for furniture in furniture_list:
                    if furniture['uid'] == object_in_room['ref'] and furniture.__contains__('valid') and furniture['valid']:
                        # temporarily disable lamp
                        furniture_id = furniture['jid']
                        pos = np.array(object_in_room['pos'])
                        rot =  np.array(object_in_room['rot'])
                       # rot = rot[[3,0,1,2]]
                        scale =  np.array(object_in_room['scale'])
                        if not os.path.exists(FUTURE_PATH + '/models/' + furniture_id + '/raw_model.obj'):
                            continue
                        #furniture_mesh = trimesh.load(FUTURE_PATH + '/3D-FUTURE-model/' + furniture_id + '/raw_model.obj',force='mesh',process=False)

                        #vert = furniture_mesh.vertices.copy()
                        #vert = vert.astype(np.float64) * scale

                        with open(FUTURE_PATH + '/models/' + furniture_id + '/bbox.txt') as f:
                            line = f.readline().split()
                            line = [float(number) for number in line]
                            maxv = np.array(line[0:3])
                            minv = np.array(line[3:6])

                       # print(minv)
                        minv = minv * scale
                        maxv = maxv * scale
                        maxv += pos
                        minv += pos
                        ref = [0,0,1]
                        axis = np.cross(ref, rot[1:])
                        theta = np.arccos(np.dot(ref, rot[1:]))*2
                        #if axis[1] < 0: theta = -theta


                        center = (maxv + minv) / 2
                        extent = (maxv - minv)

                        translation = trimesh.transformations.translation_matrix(center)
                        angles = trimesh.transformations.euler_from_quaternion(rot)
                        #print(rot)
                        #print(angles)
                        #print(axis)
                        #print(theta)
                        rotation = rotation_matrix(axis,theta)
                        #print(rotation)
                        if(axis[1] <0) : theta = - theta
                        box = trimesh.primitives.Box(extents = maxv - minv,
                        transform = np.dot(translation,rotation))
                       
                        furniture_category = categories[model_category_data[furniture_id]]
                        if furniture_category =='Lamp':
                            print('Lamp, Ignoring...')
                            continue
                        #print(furniture_category + '--' + furniture_id)
                        cate = np.eye(len(category_class))[category_class.index(furniture_category)]
                        tried = 0
                        furn_not_moveable= False

                        if not wall_not_valid:
                                aaxis = np.array([0,0,1])
                                extent_actual = box.bounding_box.primitive.extents
                                ray_dir = np.dot(aaxis,rotation[:3,:3]).reshape(1,3) * np.array([-1,0,-1])
                                ray_dir = np.repeat(ray_dir,10,axis=0)
                                extension_dir = np.abs(np.cross(ray_dir[0],np.array([0,1,0])))
                                item_extents_axis = np.array([0.,0.,0.])
                                item_extents_axis_sub = np.array([0.,0.,0.])
                                item_extents_axis[np.argmax(np.abs(extension_dir))] = extent_actual[np.argmax(np.abs(extension_dir))]
                                item_extents_axis_sub = extent_actual[np.argmax(np.abs(ray_dir[0]))] / 2 * ray_dir[0]
                            # print(item_extents_axis)
                            # print(extent_actual[np.argmax(np.abs(extension_dir))])
                                ray_origins = np.repeat(item_extents_axis.reshape(1,3),10,axis=0) * np.array(list(np.arange(-0.5,0.5,0.1))).reshape(10,1).repeat(3,axis=1) + center + item_extents_axis_sub
                                locations,rayidx,_ = wall_ray_intersector.intersects_location(ray_origins,ray_dir)
                            # print(ray_origins)
                                print(ray_dir[0])
                            # print(extent_actual)
                            # print(extent)
                                if(len(rayidx) != 0): 
                                    dd = np.min(np.linalg.norm(locations - ray_origins[rayidx],axis=1))
                                    print(dd)
                                    if(dd <= 0.2): furn_not_moveable = True
                                else:
                                    print(center)
                                    print(wall_vert_center)
                                    # sss = trimesh.Scene()
                                    # sss.add_geometry(wall_collision_mesh)
                                    # sss.add_geometry(trimesh.primitives.Box(extents=extent,transform=trimesh.transformations.translation_matrix(center)))
                                    # print('-----------------!!!!!!!!!!!!!!!!!')
                                    # sss.show()
                        matched = mesh_matching[furniture_id].copy()

                        while True:
                            if(len(matched) == 0): break
                            tried += 1
                            if tried > 300: 
                                print('try number exceed 300, continue...')
                                break

                            try:
                        #print(furniture_id)
                                all_moveable = True
                                if choiced_list.__contains__(furniture_id):
                                    choiced = choiced_list[furniture_id]
                                else:
                                    if furn_not_moveable:
                                        print(matched)
                                        for cc in mesh_matching[furniture_id]:
                                            if cc['name'].split('/')[0] not in moveable_list:
                                                all_moveable = False
                                            else:
                                                print('removed')
                                                matched.remove(cc)
                                        print(matched)
                                        print(mesh_matching[furniture_id])
                                            
                                    if all_moveable and furn_not_moveable: break
                                    choiced_orig = random.choice(matched)
                                    choiced = choiced_orig['name'].split('/')[0]
                                    # for m in matched:
                                    #     if m['name'].split('/')[0] in moveable_list and (furniture_category == 'Table' or furniture_category == 'Storage_Furniture') and tried <= 40:
                                    #         choiced = m['name'].split('/')[0]
                                    #         #print('yyy')
                                    #         break
                                    #choiced = matched[0]['name'].split('/')[0]
                                
                                if choiced in moveable_list:
                                    moveabledata = partnetmobildataset_utils.get_moveable_box_data(choiced)
                                    for i in range(len(moveabledata)):
                                        if moveabledata[i][12] == 1 and moveabledata[i][17] == 0:
                                            raise Exception
                                        if moveabledata[i][12] == 1:
                                            main_rot_axis = np.argmax(np.abs(moveabledata[i][16:19]))
                                            xx2 = [0,1,2].remove(main_rot_axis)
                                            x1 = np.max(moveabledata[i][9:12][np.array(xx2)])
                                            x2 = np.max(box.bounding_box.primitive.extents[np.array(xx2)])
                                            if np.abs(x1 - x2) <= 0.1:
                                                raise Exception
                                        

                                choiced_partnet_json_path = '/home/yangjie/208/yangjie/scene/data/partnetdata/partnetdata_new/' + choiced + '.json'
                                obj = FrontDatasetPartNet.load_object(choiced_partnet_json_path,Tree=cates_class[Object_cates.index(furniture_category)])
                                
                                #print(choiced)
                                #root_code = encoders[furniture_category].encode_structure(obj)
                                #print(root_code)
                                #furniture_root_code_list.append(root_code.detach().numpy().flatten()[:128])
                                boxes = (obj.root.box_quats(leafs_only=True))
                                box_minmax = []
                                for bb in boxes:
                                    #print(bb)
                                    bb_center = bb[0,:3]
                                    bb_extents =bb[0,3:6]
                                    box_minmax.append((bb_center - bb_extents / 2).tolist())
                                    box_minmax.append((bb_center + bb_extents / 2).tolist())
                                    #print(box_minmax)
                                box_minmax = np.array(box_minmax)
                            # print(box_minmax)
                                #print(box_minmax)
                                box_extents = np.max(box_minmax,axis=0) - np.min(box_minmax,axis=0)
                                #print('aaaaaaaaaaaa' + str(box_extents))
                                obj_orig_extents_list.append(box_extents)
                                subbox_lengths.append(len(boxes))
                                for subbox in boxes:
                                    subbox = subbox.cpu().detach().numpy().flatten()
                                    subbox_data.append(subbox)


                                angle_list.append(theta)
                                normalize_scene.add_geometry(box,node_name = str(len(angle_list) - 1))
                                furniture_category_list.append(cate)
                                choiced_list[furniture_id] = choiced
                                
                                
                                obj_partnet_list.append(choiced)
                                print(choiced)
                                print(furniture_id)
                                print(furniture_category)
                                #draw_partnet_objects([obj])

                                break

    #                          except KeyError as ex:
    #                               print(ex)
                            except Exception as ex:
                                print('exex')
                                if(len(matched) != 0):
                                    matched.remove(choiced_orig)
                                pass
                                #print(ex)
                                #traceback.print_exc()
                            
            subbox_data = np.array(subbox_data)
            if len(furniture_category_list) == 0: continue
            scale_factor = np.max(normalize_scene.bounding_box.extents)
            #print(normalize_scene.bounding_box.extents)
            #normalize_scene = normalize_scene.scaled(1 / scale_factor)
            scene_center = (normalize_scene.bounds[0] + normalize_scene.bounds[1]) / 2
            # 3 + 3 + 1 + 5 = 12
            objs = normalize_scene.dump()
            #print((furniture_category_list))
            name_list = normalize_scene.graph.nodes_geometry
            moveable_data = []
            moveable_edges = []
            for i in range(len(objs)):
                box_i = objs[i]
                extent = box_i.primitive.extents
                center = (np.max(box_i.vertices,axis=0) + np.min(box_i.vertices,axis=0)) / 2# - scene_center
                transform = box_i.primitive.transform

                scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(transform)
                rot = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])
                cate = furniture_category_list[int(name_list[i])]
                #root_code = furniture_root_code_list[int(name_list[i])]
                rot1 = rotation_matrix([0,1,0],angles[1])
                
                extent_actual = box_i.bounding_box.primitive.extents
                cate_str = category_class[list(cate).index(1)]
                extents_orig = obj_orig_extents_list[i]
                scale_s = extent / extents_orig

                

                #print(moveable_list)
                if obj_partnet_list[int(name_list[i])] in moveable_list and (cate_str == 'Table' or cate_str == 'Storage_Furniture'):
                    try:
                   # print(obj_partnet_list[int(name_list[i])])
                        moveable_boxes = partnetmobildataset_utils.get_moveable_box_data(obj_partnet_list[int(name_list[i])])
                    except:
                        print('moveable box processing error!')
                        continue
                   # print('xxxx')
                    for box in moveable_boxes:
                        orig_center = box[0:3]
                        orig_extents = box[3:6]
                        center_tmp = box[6:9]
                        extents_tmp = box[9:12]
                        
                        parent_rotation = rotation_matrix([0,1,0],angles[1])
                        #print(orig_center) 
                        #print(center)
                        

                        moveable_feature = np.zeros(9)
                    
                        if box[12] == 0:
                            moveable_feature[0:5] = box[13:18]
                            main_dir_idx = np.argmax(moveable_feature[0:3])
                            extent_scale = extent[main_dir_idx] / orig_extents[main_dir_idx]
                            moveable_feature[3] *= scale_s[main_dir_idx]
                            moveable_feature[4] *= scale_s[main_dir_idx]
                        else:
                            moveable_feature[0:8] = box[13:21]
                            moveable_feature[0:3] =  (moveable_feature[0:3] - center_tmp)/ orig_extents * extent
                            moveable_feature[6:8] =  moveable_feature[6:8] * 3.1415926535 / 180
                           
                        center_tmp = (center_tmp - orig_center) / orig_extents * extent
                        center_tmp = np.dot(parent_rotation[:3,:3],center_tmp)
                        center_tmp = center_tmp + center
                        extents_tmp = extents_tmp / orig_extents * extent

                        if box[12] == 0:
                            moveable_data.append(np.concatenate([center_tmp,extents_tmp,np.array([angle_list[int(name_list[i])]]),[0,0,0,0,0,1,0],moveable_feature]))
                        else:
                            moveable_data.append(np.concatenate([center_tmp,extents_tmp,np.array([angle_list[int(name_list[i])]]),[0,0,0,0,0,0,1],moveable_feature]))

                        moveable_edges.append([len(objs) + len(moveable_data) - 1, i])
                        moveable_edges.append([i,len(objs) + len(moveable_data) - 1])

                data.append(np.concatenate([center,extent,np.array([angle_list[int(name_list[i])]]),cate,np.zeros(9)]))
                
                
            # Basic: A Dense Graph of the edges
            #edge_index = make_dense_edge_index(len(data))
            # Level1 : Graph of Nearby Objects and same Objects
            # type 0: Normal spatial
            # type 1: same
            # type 2: moveable
            edge_index = []
            edge_type = []
            for i in range(len(data)):
                for j in range(i, len(data)):
                    if min_dist(data[i][:10],data[j][:10]) < 0.1 and i != j:
                        edge_index.append([i,j])
                        edge_index.append([j,i])
                        edge_type.append(np.eye(len(edge_category))[edge_category.index('Spatial')])
                        edge_type.append(np.eye(len(edge_category))[edge_category.index('Spatial')])
                    if np.linalg.norm(data[i][3:6] - data[j][3:6]) < 0.001 and i != j:
                        edge_index.append([i,j])
                        edge_index.append([j,i])
                        edge_type.append(np.eye(len(edge_category))[edge_category.index('Same')])
                        edge_type.append(np.eye(len(edge_category))[edge_category.index('Same')])
            for i in range(len(moveable_edges)):
                edge_type.append(np.eye(len(edge_category))[edge_category.index('Moveable')])
            edge_index += moveable_edges
            data += moveable_data       
            data = np.array(data)
            
            #print(edge_index)
            #print(edge_type)

            edge_index = np.array(edge_index).T
            edge_type = np.array(edge_type).T
           # wall_obj = trimesh.load('H:/partvae/mesh1/'+ scenename + '--' + roomtype + '.obj.obj',process=False)
            
            ret.append(BasicSceneGraph(data,subbox_data,subbox_lengths,edge_index,edge_type,room['type'],([],[]),wall_countour,room_id))

        # for s in ret:
        #     #if s._room_type == "Bedroom":
        #     s.visualize()
        #     s.visualize2D()
        return ret

    @staticmethod
    def getGraphsfromBaselineFile(path:str):
        # cope with 3D boundingbox now
        moveable_list = os.listdir('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/')
        with open('/mnt/2/sjm_env/SceneEvolutionGraph/replacement_new_rl.json','r') as f: 
            mesh_matching = json.load(f)
        with open(path) as f:
            json_data = json.load(f)
        if json_data.__contains__('objects'):
            furniture_list = json_data['objects']
        else:
            furniture_list = []
        
    
        # 3D-FUTURE: Z+
        # partnet:Z+
        data = []
        normalize_scene = trimesh.Scene()
        furniture_category_list = []
        furniture_root_code_list = []
        angle_list = []
        choiced_list = {}
        subbox_data = []
        subbox_lengths = []
        moveable_data = []
        moveable_edges = []
        obj_partnet_list = []
        obj_orig_extents_list = []
        moveable_data = []
        data = []
        wall_countour = np.array(json_data['floor']['verts'][:1024])
        #wall_countour = wall_countour[:,[0,2,1]]

        # wall_vert_center = (np.max(wall_countour,axis=0) + np.min(wall_countour,axis=0)) / 2
           

        # wall_countour2 = (wall_countour - wall_vert_center) * 1.05 + wall_vert_center
        # wall_countour_outer = (wall_countour2 - wall_vert_center) * 6 + wall_vert_center
        # wall_countour_outer = np.concatenate([wall_countour_outer,wall_countour_outer[0].reshape(1,-1)],axis=0)
        # wall_not_valid =False
        # try:
        #     poly = Polygon(wall_countour_outer[:,[0,2]],[np.concatenate([wall_countour2,wall_countour2[0].reshape(1,-1)],axis=0)[:,[0,2]]])
        #     wall_collision_mesh = trimesh.creation.extrude_polygon(poly,20)
        #     trans_toorig =  trimesh.transformations.translation_matrix([0,4,0])
        #     rot_yz = rotation_matrix(np.array([1,0,0]),3.1415926535 / 2)
        #     wall_collision_mesh = wall_collision_mesh.apply_transform(np.dot(trans_toorig,rot_yz))
        #     wall_ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(wall_collision_mesh)

        # except:
        #     # p1 = trimesh.load_path(wall_countour)
        #     # p2 = trimesh.load_path(wall_countour_outer)
        #     # p1.show()
        #     # p2.show()
        #     wall_not_valid =True
        # # patch = PolygonPatch(poly)
        # #  ax.add_patch(patch)

        # plt.show()

        ss = trimesh.Scene()
        fur_idx = 0
        for furniture in furniture_list:
                        # temporarily disable lamp
        # v, f = load_obj(objfile)
        # v = normalize(v)
       
        # v = v*np.array(box_para[1:4])   # scale
        # v_rot = r2.apply(v)             # rotation
        # v_rot += np.array(box_para[4:]) # center
        # all_f += (f + len(all_v)).tolist()
        # all_v += v_rot.tolist()
            # print(minv)
            modelid = furniture['model_id']
            choiced = modelid.split('/')[0]
           # print(modelid)
            box_para = np.array(furniture['box'])
           # print(r1)
            
            #model = trimesh.load()
            sstmp = trimesh.Scene()
            choiced_partnet_json_path = '/home/yangjie/208/yangjie/scene/data/partnetdata/partnetdata_new/' + choiced + '.json'
            obj = FrontDatasetPartNet.load_object(choiced_partnet_json_path,Tree=cates_class[2])
            boxes = (obj.root.box_quats(leafs_only=True))
            for bb in boxes:
                #print(bb)
                bb_np = bb.numpy()
                bb_center = bb_np[0,:3] 
                bb_extents = bb_np[0,3:6] 

                #print(bb_np)
                rotation = trimesh.transformations.quaternion_matrix(bb_np[0,6:10] )
                translation = trimesh.transformations.translation_matrix(bb_center)
                b = trimesh.primitives.Box(extents=bb_extents.reshape(3,),transform=np.dot(translation,rotation))
                sstmp.add_geometry(b)
                # box_minmax.append((bb_center - bb_extents / 2).tolist())
                # box_minmax.append((bb_center + bb_extents / 2).tolist())
                #print(box_minmax)
            extent = np.abs((sstmp.bounds[0] - sstmp.bounds[1]))
            scale = np.zeros((4,4))
            scale[0,0] = box_para[3] / extent[0]
            scale[1,1] = box_para[2] / extent[1]
            scale[2,2] = box_para[1] / extent[2]
            scale[3,3] = 1
         # print(box_para)
           
            r1 = trimesh.transformations.euler_matrix(0, box_para[0],0)
            r2 =trimesh.transformations.euler_matrix(0, 3.1415926/2,0)
            t1 =  trimesh.transformations.translation_matrix(box_para[[4,5,6]])
            matrix = np.dot(np.dot(np.dot(t1,r1),r2),scale)
           

            
            box_minmax = []

            center = (sstmp.bounds[0] + sstmp.bounds[1]) / 2
            extent = box_para[[3,2,1]]

            for bb in boxes:
               # print(bb)
                bb_np = bb.numpy()
                bb_center = bb_np[0,:3] 
                bb_extents = bb_np[0,3:6] 
                bb_np[0,:3] -= center
               # print(bb_np)
                rotation = trimesh.transformations.quaternion_matrix(bb_np[0,6:10] )
                translation = trimesh.transformations.translation_matrix(bb_center - center)
                b = trimesh.primitives.Box(extents=bb_extents.reshape(3,),transform=np.dot(matrix,np.dot(translation,rotation)))
                #ss.add_geometry()
                subbox_data.append(bb_np.flatten())
                # box_minmax.append((bb_center - bb_extents / 2).tolist())
                # box_minmax.append((bb_center + bb_extents / 2).tolist())
                #print(box_minmax)
            subbox_lengths.append(len(boxes))
            cate = np.eye(len(category_class))[category_class.index('Table')]
            # box_minmax = np.array(box_minmax)
            # box_extents = np.max(box_minmax,axis=0) - np.min(box_minmax,axis=0)
            data.append(np.concatenate([box_para[[4,5,6]],extent,np.array([3.1415926535 / 2 + box_para[0]]),cate,np.zeros(9)]))
            angle = 3.1415926535 / 2 + box_para[0]
            scale_coeff =  box_para[[3,2,1]]
            if choiced in moveable_list:
                moveable_boxes = partnetmobildataset_utils.get_moveable_box_data(choiced)
                for box in moveable_boxes:
                    orig_center = box[0:3]
                    orig_extents = box[3:6]
                    center_tmp = box[6:9]
                    extents_tmp = box[9:12]
                    
                    parent_rotation = rotation_matrix([0,1,0],angle)
                    #print(orig_center) 
                    #print(center)
                    

                    moveable_feature = np.zeros(9)
                
                    if box[12] == 0:
                        moveable_feature[0:5] = box[13:18]
                        main_dir_idx = np.argmax(moveable_feature[0:3])
                        extent_scale = extent[main_dir_idx] / orig_extents[main_dir_idx]
                        moveable_feature[3] *= scale_coeff[main_dir_idx]
                        moveable_feature[4] *= scale_coeff[main_dir_idx]
                    else:
                        moveable_feature[0:8] = box[13:21]
                        moveable_feature[0:3] =  (moveable_feature[0:3] - center_tmp)* scale_coeff
                        moveable_feature[6:8] =  moveable_feature[6:8] * 3.1415926535 / 180
                        
                    center_tmp = (center_tmp - orig_center) / orig_extents * extent
                    center_tmp = np.dot(parent_rotation[:3,:3],center_tmp)
                    center_tmp = center_tmp + box_para[[4,5,6]]
                    extents_tmp = extents_tmp / orig_extents * extent

                    if box[12] == 0:
                        moveable_data.append(np.concatenate([center_tmp,extents_tmp,np.array([3.1415926535 / 2 + box_para[0]]),[0,0,0,0,0,1,0],moveable_feature]))
                    else:
                        moveable_data.append(np.concatenate([center_tmp,extents_tmp,np.array([3.1415926535 / 2 + box_para[0]]),[0,0,0,0,0,0,1],moveable_feature]))

                    moveable_edges.append([len(furniture_list) + len(moveable_data) - 1, fur_idx])
                    moveable_edges.append([fur_idx,len(furniture_list) + len(moveable_data) - 1])
            fur_idx += 1
            #     data.append(np.concatenate([center,extent,np.array([angle_list[int(name_list[i])]]),cate,np.zeros(9)]))

            # type 2: moveable
        edge_index = []
        edge_type = []
        for i in range(len(data)):
            for j in range(i, len(data)):
                if min_dist(data[i][:10],data[j][:10]) < 0.1 and i != j:
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                    edge_type.append(np.eye(len(edge_category))[edge_category.index('Spatial')])
                    edge_type.append(np.eye(len(edge_category))[edge_category.index('Spatial')])
                if np.linalg.norm(data[i][3:6] - data[j][3:6]) < 0.001 and i != j:
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                    edge_type.append(np.eye(len(edge_category))[edge_category.index('Same')])
                    edge_type.append(np.eye(len(edge_category))[edge_category.index('Same')])
        for i in range(len(moveable_edges)):
            edge_type.append(np.eye(len(edge_category))[edge_category.index('Moveable')])
        edge_index += moveable_edges
        data += moveable_data       
        data = np.array(data)
            
        #     #print(edge_index)
        #     #print(edge_type)

        edge_index = np.array(edge_index).T
        edge_type = np.array(edge_type).T
        # # wall_obj = trimesh.load('H:/partvae/mesh1/'+ scenename + '--' + roomtype + '.obj.obj',process=False)
            
        return BasicSceneGraph(data,subbox_data,subbox_lengths,edge_index,edge_type,' ',([],[]),wall_countour,path)

            # # for s in ret:
            # #     #if s._room_type == "Bedroom":
            # #     s.visualize()
            # #     s.visualize2D()
            # return ret
       # ss.show()
            
if __name__=="__main__":
    graphs = []
    idx =0 
    flist = os.listdir('/home/yangjie/208/data4T/yangjie/Sync2Gen/optimization/vis/living/vaef_lr0001_w00001_B64/EXP_NAME_1')
    for idx in range(100):
        ff = str(idx).zfill(4) + '_sync.json'
           # idx = int(ff.split('_')[0])
            #idx += 1
           # if idx >= 100: continue
        print(ff)
        gg = BasicSceneGraph.getGraphsfromBaselineFile('/home/yangjie/208/data4T/yangjie/Sync2Gen/optimization/vis/living/vaef_lr0001_w00001_B64/EXP_NAME_1/' + ff)
        #gg.visualize()
        # gg.visualize2D()
        # print(gg)
        graphs.append(gg)

    print('aaaa')
    with open('./GraphAdjust/baseline_sync_livingroom.pkl','wb') as f:
        pickle.dump(graphs,f) 
    for graph in graphs:
        graph.visualize()
        graph.visualize2D()
    # with open('complete_data.pkl','rb') as f:
    #     loaded = pickle.load(f)
    # graph = BasicSceneGraph(loaded['data'][11166],loaded['edge_index'][11166],'type')
    # graph.visualize()