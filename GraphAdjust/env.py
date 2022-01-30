import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + '/../')
import numpy as np
import torch
from common import *
import trimesh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx
import pickle
from utils import load_pts, transform_pc1
import chamfer_distance
import torch
from structurenet.code.model_structure_net_box import RecursiveEncoder,RecursiveDecoder
from structurenet.code.data import SceneTree,FrontDatasetPartNet
import random
from treebase import Chair, Table, Bed, Lamp, Storage_Furniture
from collections import namedtuple
import math
import partnetmobildataset_utils
import traceback
import trimesh.sample

ACTION_SIZE_PER_ITEM = 4
ITEM_MAX_SIZE = 20

class Config():
  def __setitem__(self,k,v):
    self.__setattr__(k,v)
  def __getitem__(self,k):
    try:
      return self.__getattribute__(k)
    except AttributeError:
      return None
unit_cube = load_pts('../cube.pts')

encoders = {}
decoders = {}

Object_cates = ['Chair', 'Table', 'Bed','Lamp' ,'Storage_Furniture']
cates_class = [Chair, Table, Bed,Lamp, Storage_Furniture]

moveable_list = os.listdir('B:\\partnet-mobility-v0_2\\dataset')

PartTrees = {}
for id, cate_id in enumerate(Object_cates):
    cur_Tree = cates_class[id]
    cur_Tree.load_category_info(cate_id, cur_Tree)


for cat in category_class:
    if cat.find('Moveable') != -1 : continue

    config = torch.load(BASE_DIR + '/../' + 'structurenet/data/models/PartNetBox_vae_' + cat + '/conf.pth')
    #print(config)
    #print(cat)
    encoders[cat] = RecursiveEncoder(config,variational=True,Tree=cates_class[Object_cates.index(cat)])
    encoders[cat].load_state_dict(torch.load(BASE_DIR + '/../' + 'structurenet/data/models/PartNetBox_vae_' + cat + '/net_encoder.pth'), strict=True)
    decoders[cat] = RecursiveDecoder(config,Tree=cates_class[Object_cates.index(cat)])
    decoders[cat].load_state_dict(torch.load(BASE_DIR + '/../' + 'structurenet/data/models/PartNetBox_vae_' + cat + '/net_decoder.pth'), strict=True)

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

class ENV:
    def __init__(self,graph):
        self.done = False
        for i in range(graph._data.shape[0]):
            if graph._data[i,12] == 1 or graph._data[i,13] == 1: break
        data_tmp = graph._data[0:i]
        self.item_count_real = data_tmp.shape[0]
        self.data = data_tmp.copy()
        self.data_orig = data_tmp.copy()
        self.box_collection = []
        self.point_collection = []
        self.collision_state = np.zeros(int(ITEM_MAX_SIZE * (ITEM_MAX_SIZE - 1) / 2))
        for object_index in range(np.size(data_tmp,0)):
            object_descriptor = data_tmp[object_index]
            
            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]
            root_code = torch.tensor(object_descriptor[14:142]).float().reshape(1,128)
        

            if category_str.find('Moveable') == -1:
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
                    mesh_tmp = scene_tmp.dump(concatenate=True)

                    self.point_collection.append(trimesh.sample.sample_surface(mesh_tmp,500)[0])
                    self.box_collection.append(scene_tmp)
        self.point_collection = np.array(self.point_collection)
        self.getboxcollision()
            #self.point_collection = self.point_collection.reshape(-1,3 * 500)

    def reset(self):
        self.data = self.data_orig.copy()
        self.getboxcollision()
    
    def get_state(self):
        #return np.concatenate([self.data[:,0:7],self.point_collection,axis=1) # 6 x (7 + 1500)
        #print(self.point_collection[0,0,:])
        #ret = self.point_collection.reshape(-1,3)
        #print(self.point_collection[0,:])
        #ret = ret.T.reshape(1,3,-1)
        #print(ret[0,:,0])
        #return ret
        tmp = np.zeros((20,7))
        tmp[:self.data.shape[0],:] = self.data[:,0:7]
        return np.concatenate([tmp.flatten(),self.collision_state]).reshape(1,-1)
    def step(self, action):
        #print(action)
        item = int(action / ACTION_SIZE_PER_ITEM)
        if item >= self.item_count_real: return float(-100),False
        direction = int(action % ACTION_SIZE_PER_ITEM)

        collisions_orig = self.getboxcollision()
        #print(self.point_collection.shape)
        if direction == 0: 
            self.data[item,0] += 0.1
            self.point_collection[item,:,0] += 0.1
        if direction == 1:
            self.data[item,0] -= 0.1
            self.point_collection[item,:,0] -= 0.1
        if direction == 2:
            self.data[item,2] += 0.1
            self.point_collection[item,:,2] += 0.1
        if direction == 3: 
            self.data[item,2] -= 0.1
            self.point_collection[item,:,2] -= 0.1

        collisions = self.getboxcollision()

        # base reward = -1
        reward = -1

        if collisions >= collisions_orig: reward = -collisions * 5
        elif collisions < collisions_orig: reward = 30
        #else: reward = -collisions 
        #reward = -collisions 
        # for i in range(self.data.shape[0]):
        #     if np.abs(self.data[i,0]) >= 2.0: reward += -10
        #     if np.abs(self.data[i,2]) >= 2.0: reward += -10

        tmp = np.abs(self.data) > 2.0
        reward += tmp[:,[0,2]].sum() * -10
        done = False
        if collisions == 0: 
            reward = 500
            done = True
            self.done = True

        return float(reward),done


# return number of collision
    def getboxcollision(self):
        self.collision_state = np.zeros(int(ITEM_MAX_SIZE * (ITEM_MAX_SIZE - 1) / 2))
        ret = 0
        mgrs = []

        index = 0
        for scene in self.box_collection:
            scene_tmp = scene.copy()
            object_descriptor = self.data[index]
            bounds = (scene_tmp.bounds)
            tmp_extents = bounds[1,:] - bounds[0,:]
            tmp_center = (bounds[1,:] + bounds[0,:] ) / 2
            scene_tmp.apply_transform(trimesh.transformations.translation_matrix(-tmp_center))
            scene_tmp.apply_transform(trimesh.transformations.compose_matrix(scale=(object_descriptor[3:6]) / (tmp_extents)))
            scene_tmp.apply_transform(rotation_matrix(np.array([0,1,0]),object_descriptor[6]))
            scene_tmp.apply_transform(trimesh.transformations.translation_matrix(object_descriptor[0:3]))

            collision_mgr,_ = trimesh.collision.scene_to_collision(scene_tmp)
            mgrs.append(collision_mgr)
            index += 1

        index = 0
        for i in range(len(mgrs)):
            for j in range(i+1, len(mgrs)):
                is_collision= mgrs[i].in_collision_other(mgrs[j])
                if is_collision:
                    self.collision_state[index] = 1
                    ret += 1
                index += 1
        return ret


    def visualize(self):
        scene = trimesh.Scene()
        for object_index in range(np.size(self.data,0)):
            #if object_index != 3 and object_index != 4:continue
            object_descriptor = self.data[object_index]
            
            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]


            root_code = torch.tensor(object_descriptor[14:142]).float().reshape(1,128)
            #print(root_code.shape)
            if category_str.find('Moveable') == -1:
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
                scene_tmp.apply_transform(trimesh.transformations.translation_matrix(-tmp_center))
                #print(object_descriptor[3:6] / tmp_extents)
                scene_tmp.apply_transform(trimesh.transformations.compose_matrix(scale=(object_descriptor[3:6]) / (tmp_extents)))
                #print(object_descriptor[6])
                scene_tmp.apply_transform(rotation_matrix(np.array([0,1,0]),object_descriptor[6]))
                #print(object_descriptor[6:10])
                scene_tmp.apply_transform(trimesh.transformations.translation_matrix(object_descriptor[0:3]))
                #print(trimesh.transformations.euler_from_quaternion(object_descriptor[6:10]))
                scene.add_geometry(scene_tmp)
        scene.show()

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