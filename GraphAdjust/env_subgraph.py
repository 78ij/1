import json
import os
import sys
import subprocess
import shlex
import queue
import threading
import time
from scipy.special import entr
from fastdist import fastdist
import struct
import scipy.special
import signal
from pyquaternion import Quaternion
import pyrender
from planning import *
#from sysv_ipc import *
import cv2 as cv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + '/../')
import numpy as np
import torch
from common import *
import trimesh
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import networkx
import pickle
from utils import load_pts, transform_pc1
#import chamfer_distance
import torch
#from structurenet.code.model_structure_net_box import RecursiveEncoder,RecursiveDecoder
#from structurenet.code.data import SceneTree,FrontDatasetPartNet
import random
#from treebase import Chair, Table, Bed, Lamp, Storage_Furniture
from collections import namedtuple
import math
import partnetmobildataset_utils
import traceback
import trimesh.sample
import subprocess
#from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point

from descartes.patch import PolygonPatch

from matplotlib.path import Path
import cv2
category_class = [
    "Storage_Furniture", 
    "Table",
    "Chair", 
    "Bed", 
   # "Sofa",
    "Lamp",
    "Moveable_Slider",
    "Moveable_Revolute"
]

def get_process_id(name):
    """Return process ids found by (partial) name or regex.

    >>> get_process_id('kthreadd')
    [2]
    >>> get_process_id('watchdog')
    [10, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61]  # ymmv
    >>> get_process_id('non-existent process')
    []
    """
    child = subprocess.Popen(['pgrep', '-f', name], stdout=subprocess.PIPE, shell=False)
    response = child.communicate()[0]
    return [int(pid) for pid in response.split()]
from RRT.obstacles         import Obstacles
from RRT.ImageGenerator    import ImageGenerator
from RRT.utilities         import get_obstacle_course, get_start_and_goal

from RRT.unidirectionalrrt import run as RRT
from RRT.bidirectionalrrt  import run as BRRT
from RRT.extra_credit      import run as EXTRA
from RRT.KDTree import *


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

#encoders = {}
#decoders = {}

# Object_cates = ['Chair', 'Table', 'Bed','Lamp' ,'Storage_Furniture']
# cates_class = [Chair, Table, Bed,Lamp, Storage_Furniture]


# PartTrees = {}
# for id, cate_id in enumerate(Object_cates):
#     cur_Tree = cates_class[id]
#     cur_Tree.load_category_info(cate_id, cur_Tree)


# for cat in category_class:
#     if cat.find('Moveable') != -1 : continue

#     config = torch.load(BASE_DIR + '/../' + 'structurenet/data/models/PartNetBox_vae_' + cat + '/conf.pth')
#     #print(config)
#     #print(cat)
#     encoders[cat] = RecursiveEncoder(config,variational=True,Tree=cates_class[Object_cates.index(cat)])
#     encoders[cat].load_state_dict(torch.load(BASE_DIR + '/../' + 'structurenet/data/models/PartNetBox_vae_' + cat + '/net_encoder.pth'), strict=True)
#     decoders[cat] = RecursiveDecoder(config,Tree=cates_class[Object_cates.index(cat)])
#     decoders[cat].load_state_dict(torch.load(BASE_DIR + '/../' + 'structurenet/data/models/PartNetBox_vae_' + cat + '/net_decoder.pth'), strict=True)

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

def get_three_moveable_boxes(object_descriptor):
    #print(1)
    print(object_descriptor)
    center = object_descriptor[0:3]
    extents = object_descriptor[3:6]    
    ret = []
    #print(object_descriptor)
    #print(object_descriptor[14:22])
    category_onehot = list(object_descriptor[7:])
    category_str = category_class[category_onehot.index(1)]
    if category_str == 'Moveable_Slider':
        translation = trimesh.transformations.translation_matrix(center)
        rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])

        slide_direction = object_descriptor[14:17]

        #slide_direction = np.dot(slide_direction,rotation[0:3,0:3])
        slide_direction = slide_direction / np.linalg.norm(slide_direction)
        slide_direction = np.dot(slide_direction,rotation[:3,:3])
        translation_1 = trimesh.transformations.translation_matrix(slide_direction * object_descriptor[17])
        translation_2 = trimesh.transformations.translation_matrix(slide_direction * object_descriptor[18])
        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(np.dot(translation,translation_1),rotation))
        
        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(b.primitive.transform)

        center2 = trans
        extents2 = extents
        quat2 = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])
        object_descriptor2 = np.concatenate([center2,extents2,quat2,np.array(1).reshape(1)])
        ret.append(object_descriptor2)
        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(np.dot(translation,translation_2),rotation))
        b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
        
        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(b.primitive.transform)

        center2 = trans
        extents2 = extents
        quat2 = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])

        object_descriptor3 = np.concatenate([center2,extents2,quat2,np.array(1).reshape(1)])
        ret.append(object_descriptor3)
    else:
        translation = trimesh.transformations.translation_matrix(center)
        rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])

        rotation_orig = object_descriptor[14:17]

        rotation_direction = object_descriptor[17:20]
        #print(rotation_direction)
        # rotation_orig = np.dot(rotation_orig,rotation[0:3,0:3])
        #rotation_orig = rotation_orig / np.linalg.norm(rotation_orig)
        
        #print(rotation_orig)
        # print(rotation_direction)
        #print(object_descriptor)

        rotation_1 =  trimesh.transformations.rotation_matrix(object_descriptor[20], rotation_direction,rotation_orig)
        
        rotation_2 =  trimesh.transformations.rotation_matrix(object_descriptor[21] , rotation_direction,rotation_orig)

        rotation_3 =  trimesh.transformations.rotation_matrix((object_descriptor[20] + object_descriptor[21]) / 2, rotation_direction,rotation_orig)


        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,rotation_1)))
        b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)

        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(b.primitive.transform)
      #  print(scale, shear, angles, trans, persp)
        center2 = trans
        extents2 = extents
        quat2 = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])

        object_descriptor2 = np.concatenate([center2,extents2,quat2,np.array(1).reshape(1)])
        ret.append(object_descriptor2)
      #  print(object_descriptor2)
        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,rotation_2)))
        b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)

        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(b.primitive.transform)
     #   print(scale, shear, angles, trans, persp)
        center2 =  trans
        extents2 = extents
        quat2 = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])

        object_descriptor2 = np.concatenate([center2,extents2,quat2,np.array(1).reshape(1)])
        ret.append(object_descriptor2)
     #   print(object_descriptor2)
        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(rotation,rotation_3)))
        b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)

        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(b.primitive.transform)
       # print(scale, shear, angles, trans, persp)
        center2 = trans
        extents2 = extents
        quat2 = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])

        object_descriptor2 = np.concatenate([center2,extents2,quat2,np.array(1).reshape(1)])
        ret.append(object_descriptor2)
       # print(object_descriptor2)
    return ret
# command = 'source /home/sunjiamu/.bashrc;source /mnt/2/sjm_env/ws_moveit/devel/setup.bash;export PATH=/usr/bin:$PATH;python --version;roslaunch moveit_tutorials motion_planning_api_tutorial.launch my_args:="file=/mnt/2/sjm_env/scene_descriptor"'

# process = subprocess.Popen(command,
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                     bufsize=-1,
#                     executable='/bin/bash',
#                     shell=True,close_fds=True)
# key =  ftok("/mnt/2/sjm_env/ws_moveit/src/moveit_tutorials/doc/motion_planning_api/fraction",123)
# while True:
#     try:
#         queue = MessageQueue(key)
#     except:
#         continue
#     break
# time.sleep(5)
# pid = get_process_id('scene_descriptor __name:=motion_planning_api_tutorial __log:')[0]

class ENV:
    def __init__(self,graph):
        self.file_path = graph.room_id
        wall_countour = graph.wall_countour
        self.wall_vert_center = (np.max(wall_countour,axis=0) + np.min(wall_countour,axis=0)) / 2
        wall_vert_center = (np.max(wall_countour,axis=0) + np.min(wall_countour,axis=0)) / 2
        wall_countour = (wall_countour - wall_vert_center) * 1.05 + wall_vert_center

        self.wall_countour = (graph.wall_countour - wall_vert_center) * 1.15 + wall_vert_center
        wall_countour_center = (np.max(self.wall_countour,axis=0) + np.min(self.wall_countour,axis=0)) / 2
        wall_countour_outer = (self.wall_countour - wall_countour_center) * 6 + wall_countour_center
        wall_countour_outer = np.concatenate([wall_countour_outer,wall_countour_outer[0].reshape(1,-1)],axis=0)
        poly = Polygon(wall_countour_outer[:,[0,2]],[np.concatenate([self.wall_countour,self.wall_countour[0].reshape(1,-1)],axis=0)[:,[0,2]]])
        self.wall_contour_poly = Polygon(np.concatenate([self.wall_countour,self.wall_countour[0].reshape(1,-1)],axis=0)[:,[0,2]])
        
        # fig = plt.figure(1, dpi=90)

        #ax = fig.add_subplot(121)
        #patch = PolygonPatch(self.wall_contour_poly)
        #ax.add_patch(patch)
        #plt.autoscale(enable=True)
       # plt.show()
        self.wall_collision_mesh = trimesh.creation.extrude_polygon(poly,6)
      #  trans_tocenter =  trimesh.transformations.translation_matrix([-wall_countour_center[0],-wall_countour_center[2],0])
        trans_toorig =  trimesh.transformations.translation_matrix([0,4,0])
        rot_yz = rotation_matrix(np.array([1,0,0]),3.1415926535 / 2)
        self.wall_collision_mesh = self.wall_collision_mesh.apply_transform(np.dot(trans_toorig,rot_yz))
        cent2 = (np.max(self.wall_collision_mesh.vertices,axis=0) + np.min(self.wall_collision_mesh.vertices,axis=0)) / 2
        cent2[1] = 0
        self.wall_collision_mesh.apply_transform(trimesh.transformations.translation_matrix(-cent2))
        wall_vert_center[1] = 0
        self.wall_collision_mesh.apply_transform(trimesh.transformations.translation_matrix(wall_vert_center))
        #print('111111111111111111111111111111111111111')
        #print((np.max(self.wall_collision_mesh.vertices,axis=0) + np.min(self.wall_collision_mesh.vertices,axis=0)) / 2)
       # print(wall_vert_center)
        self.wall_ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.wall_collision_mesh)
       # self.wall_collision_mesh.show()
        self.wall_collider = trimesh.collision.CollisionManager()
        self.wall_collider.add_object('',self.wall_collision_mesh)
        wall_tmp = []
        for i in range(self.wall_countour.shape[0]):
            wall_tmp.append((self.wall_countour[i,0],self.wall_countour[i,2]))
        self.wall_polygon = Polygon(wall_tmp)
        
        self.done = False
        self.item_count_real = 0
        if graph._edge_index.size == 0:
            raise Exception()
        #print(graph._data[:,:15])
        for i in range(graph._data.shape[0]):
            if graph._data[i,12] == 1 or graph._data[i,13] == 1: break
        if i == graph._data.shape[0] - 1 and not (graph._data[i,12] == 1 or graph._data[i,13] == 1) : i += 1
        self.item_count_real = i
        #print(self.item_count_real)
        self.graph = graph
        self.edge_index = graph._edge_index
        self.edge_type = graph._edge_type
        #print(graph._data[:,:15])
        # print(self.edge_index)
        # print(self.edge_type)
        # print(self.item_count_real)
        print('1111111111111111111111111111111111111111')
        for i in range(self.edge_type.shape[1]):
            if self.edge_type[2,i] == 1: break
        if i == self.edge_type.shape[1] - 1 and not self.edge_type[2,i] == 1: i += 1
        self.edge_index = graph._edge_index[:,:i]
        self.edge_type = graph._edge_type[:,:i]
        
        moveable_dict = {}
        for i in range(graph._edge_type.shape[1]):
            if graph._edge_type[2,i] == 1:
                edgefrom = graph._edge_index[0,i]
                edgeto = graph._edge_index[1,i]
                if edgefrom < self.item_count_real: 
                    obj = edgefrom
                    moveable = edgeto
                else:
                    obj = edgeto
                    moveable = edgefrom
                if not moveable_dict.__contains__(obj):
                    moveable_dict[obj] = [moveable]
                else:
                    if moveable in moveable_dict[obj]:continue
                    moveable_dict[obj].append(moveable)
        if len(moveable_dict) == 0: raise Exception()
       # print(moveable_dict)
       # print(self.edge_index.shape)
       # print(self.edge_type.shape)
        data_tmp = graph._data
        self.data = data_tmp.copy()
        self.data_orig = data_tmp.copy()
        self.box_collection = []
        self.point_collection = []
        self.moveable_dict = moveable_dict
        self.subbox_data = []
        self.subbox_lengths = []
        self.subbox_edge_indices =[[0,0],[1,1]]
        self.subbox_edge_type =[[0,0,0],[0,0,0]]
        self.collision_state = np.zeros(int(ITEM_MAX_SIZE * (ITEM_MAX_SIZE - 1) / 2))
        self.moveable_boxes = []
        self.moveable_box_mapping = {}
        subbox_length_sum = [0]
        s_=0
        for xx in range(self.data.shape[0]):
            if np.sum(self.data[xx,3:6]) >= 20: raise Exception()
        for length in graph._subbox_length:
            s_+= length
            subbox_length_sum.append(s_)
        print('1111111111111111111111111111111111111111')

        for object_index in range(np.size(data_tmp,0)):
            object_descriptor = data_tmp[object_index]
            
            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]
           # root_code = torch.tensor(object_descriptor[14:142]).float().reshape(1,128)

            box_tmp = []
            if category_str.find('Moveable') == -1:
                scene_tmp = trimesh.Scene()

                boxes = graph._subbox_data[subbox_length_sum[object_index]:subbox_length_sum[object_index+1]]
                moveable_boxes = []
                if moveable_dict.__contains__(object_index):
                    for moveable_idx in moveable_dict[object_index]:
                        moveable_boxes += get_three_moveable_boxes(graph._data[moveable_idx])
                        #print('len ' + str(len(moveable_dict[object_index])))
                    self.subbox_lengths.append(len(boxes) + len(moveable_boxes))
                else:

                    self.subbox_lengths.append(len(boxes))
                #print(moveable_boxes)
                for box in boxes:
                    box = box.flatten()
                    center = box[0:3]
                    extents = box[3:6]
                    rot = box[6:10]
                    

                    translation = trimesh.transformations.translation_matrix(center)
                    rotation = trimesh.transformations.quaternion_matrix(rot)
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
                    b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
                    scene_tmp.add_geometry(b)

                    #scene_tmp.rezero()
                for box in boxes:
                    box = box.flatten()
                    center = box[0:3]
                    extents = box[3:6]
                    if extents[0]< 0.0000001: extents[0] = 0.0000001
                    if extents[1]< 0.0000001: extents[1] = 0.0000001
                    if extents[2]< 0.0000001: extents[2] = 0.0000001
                    rot = box[6:10]

                    translation = trimesh.transformations.translation_matrix(center)
                    rotation = trimesh.transformations.quaternion_matrix(rot)
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))

                    bounds = (scene_tmp.bounds)
                    tmp_extents = bounds[1,:] - bounds[0,:]
                    #print((object_descriptor[3:6]) / (tmp_extents))
                    tmp_center = (bounds[1,:] + bounds[0,:] ) / 2
                    b.apply_transform(trimesh.transformations.translation_matrix(-tmp_center))
                    b.apply_transform(trimesh.transformations.compose_matrix(scale=(object_descriptor[3:6]) / (tmp_extents)))
                    b.apply_transform(rotation_matrix(np.array([0,1,0]),object_descriptor[6]))
                    box_tmp.append(b.copy())
                    b.apply_transform(trimesh.transformations.translation_matrix(object_descriptor[0:3]))
                    

                    extents_final = b.bounding_box_oriented.primitive.extents
                    transform_final = b.bounding_box_oriented.primitive.transform
                    scale, shear, angles, translate, perspective = trimesh.transformations.decompose_matrix(transform_final)
                    rot_final = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])
                    # print(translate)
                    # print(extents_final)
                    # print(rot_final)
                    self.subbox_data.append(np.concatenate((translate,extents_final,rot_final,np.array(0).reshape(1))))
                if len(moveable_boxes) != 0:
                    for box in moveable_boxes:
                        center = box[0:3]
                        extents = box[3:6]
                        if extents[0]< 0.0000001: extents[0] = 0.0000001
                        if extents[1]< 0.0000001: extents[1] = 0.0000001
                        if extents[2]< 0.0000001: extents[2] = 0.0000001
                        rot = box[6:10]

                        translation = trimesh.transformations.translation_matrix(center)
                        rotation = trimesh.transformations.quaternion_matrix(rot)
                        b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
                        b.apply_transform(trimesh.transformations.translation_matrix(-tmp_center))
                        scale, shear, angles, translate, perspective = trimesh.transformations.decompose_matrix(b.primitive.transform.copy())

                        b.apply_transform(trimesh.transformations.translation_matrix(-object_descriptor[0:3]))
                        #b.apply_transform(trimesh.transformations.compose_matrix(scale=(object_descriptor[3:6]) / (tmp_extents)))
                      #  b.apply_transform(rotation_matrix(np.array([0,1,0]),object_descriptor[6]))
                        box_tmp.append(b.copy())
                      #  b.apply_transform(trimesh.transformations.translation_matrix(object_descriptor[0:3]))
                        
                        rot_final = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])

                        self.subbox_data.append(np.concatenate((translate,extents,rot_final,np.array(1).reshape(1))))
                        self.moveable_boxes.append(np.concatenate((translate,extents,rot_final,np.array(1).reshape(1))))
                self.box_collection.append(box_tmp)
        self.subbox_data = np.array(self.subbox_data)
        self.moveable_boxes = np.array(self.moveable_boxes)
        self.subbox_data_orig = self.subbox_data.copy()
        self.tmp_subbox_length_sum = [0]
        sum_tmp = 0
        for l in self.subbox_lengths:
            self.tmp_subbox_length_sum.append(sum_tmp + l)
            sum_tmp += l
        
        print('1111111111111111111111111111111111111111')
        self.edge_index = []
        self.edge_type = []
        for i in range(self.item_count_real):
            for j in range(self.item_count_real):
                if i == j: continue
                dist_obj = np.linalg.norm(self.data[i,:3] - self.data[j,:3])
                dist_obj_max = (np.linalg.norm(self.data[i,3:6]) +  np.linalg.norm(self.data[j,:6])) / 2
                if dist_obj + 0.05 > dist_obj_max:continue
                self.edge_index.append([i,j])
                self.edge_type.append(self.data[i,:3] - self.data[j,:3])
                for x in range(self.tmp_subbox_length_sum[i], self.tmp_subbox_length_sum[i + 1]):
                    for y in range(self.tmp_subbox_length_sum[j], self.tmp_subbox_length_sum[j + 1]):
                        dist = np.linalg.norm(self.subbox_data[x,:3] - self.subbox_data[y,:3])
                        dist_max = (np.linalg.norm(self.subbox_data[x,3:6]) +  np.linalg.norm(self.subbox_data[y,3:6])) / 2
                        if dist > dist_max: continue
                        self.subbox_edge_indices.append([x,y]) 
                        #self.subbox_edge_type.append(self.edge_type[:,i])
                        self.subbox_edge_type.append(self.subbox_data[x,:3] - self.subbox_data[y,:3])

        self.edge_index= np.array(self.edge_index).T
        self.edge_type = np.array(self.edge_type).T
        
      
        self.subbox_edge_indices = np.array(self.subbox_edge_indices).T
        if self.subbox_edge_indices.size == 0:
            raise Exception()
        self.subbox_edge_type = np.array(self.subbox_edge_type).T

        self.mgrs = []


        index = 0
       
        for scene in self.box_collection:
            scene_x = trimesh.Scene()
            box_tmplist = []
            collision_mgr = trimesh.collision.CollisionManager()
            for box in scene:
                object_descriptor = self.data[index]
                boxt = box.copy()
                box_tmplist.append(boxt)
            mesh_concat = trimesh.util.concatenate(box_tmplist)
           # mesh_concat.show()
            #mesh_concat.show()
           # mesh_concat.apply_transform(trimesh.transformations.translation_matrix(self.data[index,0:3]))
            #print(self.moveable_boxes)
           # scene_x.add_geometry(mesh_concat)
           # scene_x.add_geometry(self.wall_collision_mesh)
           # scene_x.show()
            collision_mgr.add_object(' ',mesh_concat)
            self.mgrs.append(collision_mgr)
            index += 1
        #scene_x.show()
      #  print(self.subbox_edge_indices)
       # print(self.subbox_edge_type)
        #print(self.subbox_edge_type.shape)
        # print(self.data[:,:15])
        # print(self.subbox_lengths)
        # print(self.item_count_real)
        self.bbox_data = np.zeros((self.data.shape[0],3))
        self.sample_points = []
        print('1111111111111111111111111111111111111111')
        for i in range(self.data.shape[0]):
            object_descriptor = self.data[i]
            translation = trimesh.transformations.translation_matrix(object_descriptor[:3])
            rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])
            b = trimesh.primitives.Box(extents=self.data[i,3:6],transform=rotation)
            self.sample_points.append(b.sample(300))
            self.bbox_data[i] = b.bounding_box.primitive.extents
        self.getboxcollision()
        room_min = np.min(self.wall_countour,axis=0)[[0,2]]
        room_max = np.max(self.wall_countour,axis=0)[[0,2]]
        self.query_array = np.zeros((30,30))
        self.query_points = np.zeros((30,30,2))
        self.human_collider = trimesh.collision.CollisionManager()
        self.human_collider.add_object(' ',trimesh.load('./human.obj',force='mesh'))
        offset_point = (room_max - room_min) / 30 / 2

        for i in range(30):
            for j in range(30):
                self.query_points[i,j] = (room_min) + (room_max - room_min) * np.array([i,j]) / 30. + offset_point
                pp = Point(self.query_points[i,j,0],self.query_points[i,j,1])
                self.query_array[i,j] = self.wall_contour_poly.contains(pp)
            #self.point_collection = self.point_collection.reshape(-1,3 * 500)
        self.per_area = (room_max - room_min) / 30
        self.per_area = self.per_area[0] * self.per_area[1]
        self.floor_area = (self.query_array != 0).sum()
        self.floor_level = self.data[0,1] - (self.data[0,4]) / 2
        self.orig_fspace = self.get_free_space()


    def reset(self):
        self.subbox_data = self.subbox_data_orig.copy()
        self.data = self.data_orig.copy()
        self.subbox_edge_indices =[[0,0],[1,1]]
        self.subbox_edge_type =[[0,0,0],[0,0,0]]
        self.edge_index = []
        self.edge_type = []
        for i in range(self.item_count_real):
            for j in range(self.item_count_real):
                if i == j: continue
                dist_obj = np.linalg.norm(self.data[i,:3] - self.data[j,:3])
                dist_obj_max = (np.linalg.norm(self.data[i,3:6]) +  np.linalg.norm(self.data[j,:6])) / 2
                if dist_obj + 0.05 > dist_obj_max:continue
                self.edge_index.append([i,j])
                self.edge_type.append(self.data[i,:3] - self.data[j,:3])
                for x in range(self.tmp_subbox_length_sum[i], self.tmp_subbox_length_sum[i + 1]):
                    for y in range(self.tmp_subbox_length_sum[j], self.tmp_subbox_length_sum[j + 1]):
                        dist = np.linalg.norm(self.subbox_data[x,:3] - self.subbox_data[y,:3])
                        dist_max = (np.linalg.norm(self.subbox_data[x,3:6]) +  np.linalg.norm(self.subbox_data[y,3:6])) / 2
                        if dist > dist_max: continue
                        self.subbox_edge_indices.append([x,y]) 
                        #self.subbox_edge_type.append(self.edge_type[:,i])
                        self.subbox_edge_type.append(self.subbox_data[x,:3] - self.subbox_data[y,:3])
        self.subbox_edge_indices = np.array(self.subbox_edge_indices).T
        self.subbox_edge_type = np.array(self.subbox_edge_type).T
        self.edge_index= np.array(self.edge_index).T
        self.edge_type = np.array(self.edge_type).T
        self.getboxcollision()
    
    def get_state(self):
        #return np.concatenate([self.data[:,0:7],self.point_collection,axis=1) # 6 x (7 + 1500)
        #print(self.point_collection[0,0,:])
        #ret = self.point_collection.reshape(-1,3)
        #print(self.point_collection[0,:])
        #ret = ret.T.reshape(1,3,-1)
        #print(ret[0,:,0])
        #return ret
        #print(self.edge_index)
        wall_distances = np.zeros((self.item_count_real,4),np.float32)
        for i in range(self.item_count_real):
            for direction in range(4):
                wall_distances[i,direction] = self.get_wall_dist(i,direction)
        return  self.data[:self.item_count_real,:7].copy(),self.edge_index.copy(),self.edge_type.copy(), self.subbox_data.copy(), self.subbox_edge_indices.copy() , self.subbox_edge_type.copy() ,self.subbox_lengths.copy(), wall_distances.copy()
    def get_free_space(self):
        
        human_obj_level =  1.7568 / 2 + self.floor_level  

        # print('------------------------------------------')
        # print(human_obj.bounding_box.primitive.extents)
        # print(human_obj.bounding_box.primitive.transform)
        # print((np.max(human_obj.vertices,axis=0) + np.min(human_obj.vertices,axis=0)) / 2)
        
        query_point_2 =  self.query_array.copy()
        ss = trimesh.Scene()
        for j in range(30):
            for i in range(30):
               # if not self.query_array[i,j]: continue
                pp =np.array([0.,0.,0.])
                pp [[0,2]] = self.query_points[i,j]
                pp[1] = human_obj_level
             #   print(pp)
                translation = trimesh.transformations.translation_matrix(pp)
                rot_tmp = rotation_matrix(np.array([0,1,0]),3.1415926 / 2)
               # human_obj.apply_transform(translation)
                # ss.add_geometry(human_obj)
                # ss.add_geometry(self.wall_collision_mesh)
                # ss.show()
                # human_obj.apply_transform(np.linalg.inv(translation))
                self.human_collider.set_transform(' ',np.dot(translation,rot_tmp))

                c1 = self.wall_collider.in_collision_other(self.human_collider)
                self.human_collider.set_transform(' ',translation)
                c2 = self.wall_collider.in_collision_other(self.human_collider)
                if c1 and c2: query_point_2[i,j] = 0

                for idx in range(len(self.mgrs)):
                    object_descriptor = self.data[idx]
                  #  print(object_descriptor[0:3])
                    self.mgrs[idx].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
                    c1 = self.mgrs[idx].in_collision_other(self.human_collider)
                    self.human_collider.set_transform(' ',translation)
                    c2 = self.mgrs[idx].in_collision_other(self.human_collider)
                    if c1 and c2: query_point_2[i,j] = 0

        visited = np.zeros((30,30))
        max_result = 0
        current_result = 0
        worklist = []
        #print('-----------------------------------------------')
        for j in range(30):
            for i in range(30):
                if not query_point_2[i,j]: continue
                if visited[i,j]: continue
                worklist.append((i,j))
                visited[i,j] = 1
                current_result = 0
                current_result +=1 
                while(len(worklist) != 0):
                    tmp = worklist[0]
                    worklist = worklist[1:]
                    if tmp[0] > 0:
                        tmp1 = (tmp[0] - 1,tmp[1])
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[0] < 29:
                        tmp1 = (tmp[0] + 1,tmp[1])
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[1] > 0:
                        tmp1 = (tmp[0],tmp[1] - 1)
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[1] < 29:
                        tmp1 = (tmp[0],tmp[1] + 1)
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[0] > 0 and tmp[1] > 0:
                        tmp1 = (tmp[0] - 1,tmp[1] - 1)
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[0] > 0 and tmp[1] < 29:
                        tmp1 = (tmp[0] - 1,tmp[1] + 1)
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[0] < 29 and tmp[1] > 0:
                        tmp1 = (tmp[0] + 1,tmp[1] - 1)
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                    if tmp[0] < 14 and tmp[1] < 14:
                        tmp1 = (tmp[0] + 1,tmp[1] + 1)
                        if query_point_2[tmp1] and not visited[tmp1]: 
                            worklist.append(tmp1)
                            visited[tmp1] = 1
                            current_result += 1
                
                if max_result < current_result: max_result = current_result
        total_area = 0
        for object_index in range(np.size(self.data[:self.item_count_real],0)):

            object_descriptor = self.data[object_index]
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]
            total_area += extents[0] * extents[2]
        #print(total_area)
        #print((self.floor_area)* self.per_area)
        #print(max_result)
       # print(max_result * self.per_area / ((self.floor_area)* self.per_area - total_area))
        return max_result * self.per_area / ((self.floor_area)* self.per_area - total_area)
        
        # fig = plt.figure(1, dpi=90)

        # ax = fig.add_subplot(121)
        # patch = PolygonPatch(self.wall_contour_poly)
        # for i in range(30):
        #     for j in range(30):
        #         if query_point_2[i,j]:
        #             ax.plot(self.query_points[i,j,0],self.query_points[i,j,1],'go')
        #         else:
        #             ax.plot(self.query_points[i,j,0],self.query_points[i,j,1],'ro')
       
       

        
        #ax.add_patch(patch)

        # ax.scatter(self.wall_countour[:,0],self.wall_countour[:,2],linewidths=0.1,marker='.')
        # xmin = np.min(self.wall_countour[:,0])
        # xmax = np.max(self.wall_countour[:,0])
        # ymin = np.min(self.wall_countour[:,2])
        # ymax = np.max(self.wall_countour[:,2])
        # centers = []
        # for object_index in range(np.size(self.data[:self.item_count_real],0)):

        #     object_descriptor = self.data[object_index]
        #     center = object_descriptor[0:3]
        #     extents = object_descriptor[3:6]

        #     translation = trimesh.transformations.translation_matrix(center)
        #     rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])
        #     b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))

        #     minv = np.min(b.vertices,axis=0)
        #     maxv = np.max(b.vertices,axis=0)
    
        #     ax.add_line(Line2D([minv[0],maxv[0]],[minv[2],minv[2]]))
        #     ax.add_line(Line2D([maxv[0],maxv[0]],[minv[2],maxv[2]]))
        #     ax.add_line(Line2D([maxv[0],minv[0]],[maxv[2],maxv[2]]))
        #     ax.add_line(Line2D([minv[0],minv[0]],[maxv[2],minv[2]]))

        #     category_onehot = list(object_descriptor[7:])
        #     category_str = category_class[category_onehot.index(1)]

        #     dist = fastdist.matrix_to_matrix_distance((self.sample_points[object_index] + center)[:,[0,2]], self.wall_countour[:,[0,2]], fastdist.euclidean, "euclidean")
        #     category_str += ("_" + str(np.min(dist)))


        #     plt.text(minv[0],maxv[2],category_str)
        #     centerv = (minv + maxv) / 2
        #     centers.append(centerv)
        # for object_index in range(self.subbox_data.shape[0]):
        #     object_descriptor = self.subbox_data[object_index]
        #     if object_descriptor[-1] != 1: continue
        #     center = object_descriptor[0:3]
        #     extents = object_descriptor[3:6]
        #     print(object_descriptor)

        #     translation = trimesh.transformations.translation_matrix(center)
        #     rotation = trimesh.transformations.quaternion_matrix(object_descriptor[6:10])
        #     b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
        #     vert = b.vertices
        #     my_quaternion = Quaternion(object_descriptor[6:10])
        #     p1 = my_quaternion.rotate(np.array([extents[0] / 2,0,extents[2] / 2])) + center
        #     p2 = my_quaternion.rotate(np.array([extents[0] / 2,0,-extents[2] / 2])) + center
        #     p3 = my_quaternion.rotate(np.array([-extents[0] / 2,0,-extents[2] / 2])) + center
        #     p4 = my_quaternion.rotate(np.array([-extents[0] / 2,0,extents[2] / 2])) + center

        #     print(p1,p2,p3,p4,center)

            
        #     ax.add_line(Line2D([p1[0],p2[0]],[p1[2],p2[2]]))
        #     ax.add_line(Line2D([p2[0],p3[0]],[p2[2],p3[2]]))
        #     ax.add_line(Line2D([p3[0],p4[0]],[p3[2],p4[2]]))
        #     ax.add_line(Line2D([p4[0],p1[0]],[p4[2],p1[2]]))
        # for index in range(len(self.box_collection)):
        #     object_descriptor = self.data[index]
        #     self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        # index = 0
        # ret = 0
        # for i in range(len(self.mgrs)):
        #     for j in range(i+1, len(self.mgrs)):
        #         is_collision= self.mgrs[i].in_collision_other(self.mgrs[j])
        #         if is_collision:
        #             ax.add_line(Line2D([self.data[i,0],self.data[j,0]],[self.data[i,2],self.data[j,2]]))

        # plt.autoscale(enable=True)
        # plt.show()
    def get_state_for_mcts(self):
        wall_distances = np.zeros((self.item_count_real,4),np.float32)
        for i in range(self.item_count_real):
            for direction in range(4):
                wall_distances[i,direction] = self.get_wall_dist(i,direction)
        return self.data.copy(), self.edge_index.copy(),self.edge_type.copy(), self.subbox_data.copy(), self.subbox_edge_indices.copy() , self.subbox_edge_type.copy(),self.subbox_lengths.copy(), wall_distances.copy()
    def set_state(self,state):
        self.data, self.edge_index,self.edge_type,self.subbox_data, self.subbox_edge_indices , self.subbox_edge_type ,self.subbox_lengths,_ = state[0].copy(),state[1].copy(),state[2].copy(),state[3].copy(),state[4].copy(),state[5].copy(),state[6].copy(),state[7]
    def get_wall_dist(self,item,direction):
        
        ray_dir = np.zeros((1,3))
        item_center = self.data[item,0:3]
        if direction == 0: 
            #self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] += 0.1
           # self.data[item,0] +=0.1
            ray_dir[:,:] = np.array([1,0,0])
        if direction == 1:
           # self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] -= 0.1
           # self.data[item,0] -= 0.1
            ray_dir[:,:] = np.array([-1,0,0])
        if direction == 2:
           # self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] += 0.1
           # self.data[item,2] += 0.1
            ray_dir[:,:] = np.array([0,0,1])
        if direction == 3: 
          #  self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] -= 0.1
           # self.data[item,2] -= 0.1
            ray_dir[:,:] = np.array([0,0,-1])
     

        ray_dir = np.repeat(ray_dir,10,axis=0)
        extension_dir = np.abs(np.cross(ray_dir[0],np.array([0,1,0])))
        item_extents_axis = np.array([0,0,0])
        item_extents_axis_sub = np.array([0,0,0])
        item_extents_axis[np.argmax(extension_dir)] = self.bbox_data[item,np.argmax(extension_dir)]
        item_extents_axis_sub = self.bbox_data[item,np.argmax(np.abs(ray_dir[0]))] / 2 * ray_dir[0]
        ray_origins = np.repeat(item_extents_axis.reshape(1,3),10,axis=0) * np.array(list(np.arange(-1,1,0.2))).reshape(10,1).repeat(3,axis=1) + item_center + item_extents_axis_sub
        locations,rayidx,_ = self.wall_ray_intersector.intersects_location(ray_origins,ray_dir)
        try:
            min_dist = np.min(np.linalg.norm(locations - ray_origins[rayidx],axis=1))
        except:
            min_dist = 1000
            # sss = trimesh.Scene()
            # sss.add_geometry(self.wall_collision_mesh)
            # translation = trimesh.transformations.translation_matrix(item_center)
            # rotation = rotation_matrix(np.array([0,1,0]),self.data[item,6])
            # b = trimesh.primitives.Box(extents=self.data[item,3:6],transform=np.dot(translation,rotation))
            # sss.add_geometry(b)
            # sss.show()
        return min_dist


    def step(self, action):
        #print(action) 
        # X = self.visualize2D()
    
        # plt.imshow(X)
        # plt.show()
        #print('step!!!!!!!')
        item = int(action / ACTION_SIZE_PER_ITEM)
        #print(item,self.item_count_real)
        #print(self.data)
        if item >= self.item_count_real + 1: return float(-500),False
        direction = int(action % ACTION_SIZE_PER_ITEM)
        reward = 0
        collisions_orig,dist_orig = self.getboxcollision()
        wall_collisions_orig = self.getwallcollision()
        #print(self.point_collection.shape)
     #   print(dist_orig)
        # first try to get the wall....

        min_dist = self.get_wall_dist(item,direction)
        #print(min_dist)
        need_move = True
       # print('1' + str(reward))
        if min_dist <= 0.00001: 
            reward += -30
            need_move = False
        elif min_dist >= 0.1: min_dist = 0.1
      #  print('2' + str(reward))
        done = False
        if item == self.item_count_real:
            done = True
            reward -= 60
        elif need_move:
            if direction == 0: 
                self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] += min_dist
                self.data[item,0] +=min_dist
            if direction == 1:
                self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] -= min_dist
                self.data[item,0] -= min_dist
            if direction == 2:
                self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] += min_dist
                self.data[item,2] += min_dist
            if direction == 3: 
                self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] -= min_dist
                self.data[item,2] -= min_dist
        
     #   print('3' + str(reward))
        self.subbox_edge_indices =[[0,0],[1,1]]
        self.subbox_edge_type =[[0,0,0],[0,0,0]]
        self.edge_index = []
        self.edge_type = []
        for i in range(self.item_count_real):
            for j in range(self.item_count_real):
                if i == j: continue
                dist_obj = np.linalg.norm(self.data[i,:3] - self.data[j,:3])
                dist_obj_max = (np.linalg.norm(self.data[i,3:6]) +  np.linalg.norm(self.data[j,:6])) / 2
                if dist_obj + 0.05 > dist_obj_max:continue
                self.edge_index.append([i,j])
                self.edge_type.append(self.data[i,:3] - self.data[j,:3])
                for x in range(self.tmp_subbox_length_sum[i], self.tmp_subbox_length_sum[i + 1]):
                    for y in range(self.tmp_subbox_length_sum[j], self.tmp_subbox_length_sum[j + 1]):
                        dist = np.linalg.norm(self.subbox_data[x,:3] - self.subbox_data[y,:3])
                        dist_max = (np.linalg.norm(self.subbox_data[x,3:6]) +  np.linalg.norm(self.subbox_data[y,3:6])) / 2
                        if dist > dist_max: continue
                        self.subbox_edge_indices.append([x,y]) 
                        #self.subbox_edge_type.append(self.edge_type[:,i])
                        self.subbox_edge_type.append(self.subbox_data[x,:3] - self.subbox_data[y,:3])
        self.subbox_edge_indices = np.array(self.subbox_edge_indices).T
        self.subbox_edge_type = np.array(self.subbox_edge_type).T
        self.edge_index= np.array(self.edge_index).T
        self.edge_type = np.array(self.edge_type).T

        collisions,dist_after = self.getboxcollision()
        wall_collisions = self.getwallcollision()
     #   print(dist_after)
        fspace = self.get_free_space()
        #print('regular' + str(collisions))
        #print('wall' + str(wall_collisions))
        # base reward = -1
        robot_fraction = 0# (self.get_robot_viability() - 0.5) * 2
        #print('---------------------------  ' + str(robot_fraction))
        #entr = self.get_entropy()
        reward +=  -collisions + robot_fraction - wall_collisions * 5 + fspace * 2#+ (dist_after - dist_orig)# + #+ entr 
      #  print('4' + str(reward))
        #print( str(wall_collisions))
        sizes = self.data[item,3] + self.data[item,4] + self.data[item,5]
       # order = np.argsort(sizes).argsort()[item]
        reward -= sizes * 0.5
       # print('5' + str(reward))
        #if collisions < collisions_orig: reward += 10      #else: reward = -collisions 
       # elif collisions > collisions_orig: reward -= 10
        #print('6' + str(reward))
        dist_1 = 0
        dist_2 = 0

       

        for (k,v) in dist_after.items():
            if dist_orig.__contains__(k):
                dist_2 += dist_orig[k]
                dist_1 += dist_after[k]
        reward += (dist_1 - dist_2) * 30
        # for i in range(self.data.shape[0]):
        #     if np.abs(self.data[i,0]) >= 2.0: reward += -10
        #     if np.abs(self.data[i,2]) >= 2.0: reward += -10
     #   print('7' + str(reward))
        #tmp = np.abs(self.data) > 8.0
        #reward += tmp[:,[0,2]].sum() * -5
        if (wall_collisions_orig!= 0 or collisions_orig != 0) and  collisions == 0 and wall_collisions == 0: 
        #     reward = 150
            reward += 20
        #     self.done = True
        if wall_collisions_orig== 0 and collisions_orig == 0 and  (collisions != 0 or wall_collisions != 0):
            reward -= 30
        if collisions == 0 and wall_collisions == 0 and fspace - self.orig_fspace >= 0.05: 
            done = True
            reward += 50
    #    print('8' + str(reward))
        return float(reward),done

    def step_heru(self, action):
        #print(action) 
        # X = self.visualize2D()
    
        # plt.imshow(X)
        # plt.show()
        print('step!!!!!!!')
        item = int(action / ACTION_SIZE_PER_ITEM)
        #print(item,self.item_count_real)
        #print(self.data)
        if item >= self.item_count_real: return float(-500),False
        direction = int(action % ACTION_SIZE_PER_ITEM)
        reward = 0
        collisions_orig,dist_orig = self.getboxcollision()
        wall_collisions_orig = self.getwallcollision()
        #print(self.point_collection.shape)
     #   print(dist_orig)
        # first try to get the wall....

        min_dist = self.get_wall_dist(item,direction)
        #print(min_dist)
        need_move = True
        if min_dist <= 0.00001: 
            reward += -30
            need_move = False
        elif min_dist >= 0.1: min_dist = 0.1
        
        if need_move:
            if direction == 0: 
              self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] += min_dist
              self.data[item,0] +=min_dist
            if direction == 1:
              self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] -= min_dist
              self.data[item,0] -= min_dist
            if direction == 2:
              self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] += min_dist
              self.data[item,2] += min_dist
            if direction == 3: 
              self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] -= min_dist
              self.data[item,2] -= min_dist

        self.subbox_edge_indices =[[0,0],[1,1]]
        self.subbox_edge_type =[[0,0,0],[0,0,0]]
        self.edge_index = []
        self.edge_type = []
        for i in range(self.item_count_real):
            for j in range(self.item_count_real):
                if i == j: continue
                dist_obj = np.linalg.norm(self.data[i,:3] - self.data[j,:3])
                dist_obj_max = (np.linalg.norm(self.data[i,3:6]) +  np.linalg.norm(self.data[j,:6])) / 2
                if dist_obj + 0.05 > dist_obj_max:continue
                self.edge_index.append([i,j])
                self.edge_type.append(self.data[i,:3] - self.data[j,:3])
                for x in range(self.tmp_subbox_length_sum[i], self.tmp_subbox_length_sum[i + 1]):
                    for y in range(self.tmp_subbox_length_sum[j], self.tmp_subbox_length_sum[j + 1]):
                        dist = np.linalg.norm(self.subbox_data[x,:3] - self.subbox_data[y,:3])
                        dist_max = (np.linalg.norm(self.subbox_data[x,3:6]) +  np.linalg.norm(self.subbox_data[y,3:6])) / 2
                        if dist > dist_max: continue
                        self.subbox_edge_indices.append([x,y]) 
                        #self.subbox_edge_type.append(self.edge_type[:,i])
                        self.subbox_edge_type.append(self.subbox_data[x,:3] - self.subbox_data[y,:3])
        self.subbox_edge_indices = np.array(self.subbox_edge_indices).T
        self.subbox_edge_type = np.array(self.subbox_edge_type).T
        self.edge_index= np.array(self.edge_index).T
        self.edge_type = np.array(self.edge_type).T

        collisions,dist_after = self.getboxcollision()
        wall_collisions = self.getwallcollision()
     #   print(dist_after)
        fspace = self.get_free_space()
       # print('regular' + str(collisions))
       # print('wall' + str(wall_collisions))
        # base reward = -1
        robot_fraction = 0# (self.get_robot_viability() - 0.5) * 2
       # print('---------------------------  ' + str(robot_fraction))
        #entr = self.get_entropy()
        reward +=  -collisions *3 + robot_fraction - wall_collisions * 5 + fspace * 2#+ (dist_after - dist_orig)# + #+ entr 
        #print( str(wall_collisions))
        sizes = self.data[item,3] + self.data[item,4] + self.data[item,5]
       # order = np.argsort(sizes).argsort()[item]
        reward -= sizes * 0.1
        if collisions < collisions_orig: reward += 1      #else: reward = -collisions 
        #reward = -collisions 
        # for i in range(self.data.shape[0]):
        #     if np.abs(self.data[i,0]) >= 2.0: reward += -10
        #     if np.abs(self.data[i,2]) >= 2.0: reward += -10

        #tmp = np.abs(self.data) > 8.0
        #reward += tmp[:,[0,2]].sum() * -5
        done = False
        if collisions == 0 and wall_collisions == 0:
            done = True
            reward += 50

        return float(reward),done

    def step_random(self, item,translate):
        #print(action) 
        # X = self.visualize2D()
        wall_collisions_orig = self.getwallcollision()
        # plt.imshow(X)
        # plt.show()
        print('step!!!!!!!')
        #print(item,self.item_count_real)
        #print(self.data)
        if item >= self.item_count_real: return float(-500),False
        reward = 0
        collisions_orig,dist_orig = self.getboxcollision()
        #print(self.point_collection.shape)
     #   print(dist_orig)
        # first try to get the wall....

        self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],0] += translate[0]
        self.subbox_data[self.tmp_subbox_length_sum[item]:self.tmp_subbox_length_sum[item+1],2] += translate[1]
        self.data[item,0] += translate[0]
        self.data[item,2] += translate[1]

        self.subbox_edge_indices =[[0,0],[1,1]]
        self.subbox_edge_type =[[0,0,0],[0,0,0]]
        self.edge_index = []
        self.edge_type = []
        for i in range(self.item_count_real):
            for j in range(self.item_count_real):
                if i == j: continue
                dist_obj = np.linalg.norm(self.data[i,:3] - self.data[j,:3])
                dist_obj_max = (np.linalg.norm(self.data[i,3:6]) +  np.linalg.norm(self.data[j,:6])) / 2
                if dist_obj + 0.05 > dist_obj_max:continue
                self.edge_index.append([i,j])
                self.edge_type.append(self.data[i,:3] - self.data[j,:3])
                for x in range(self.tmp_subbox_length_sum[i], self.tmp_subbox_length_sum[i + 1]):
                    for y in range(self.tmp_subbox_length_sum[j], self.tmp_subbox_length_sum[j + 1]):
                        dist = np.linalg.norm(self.subbox_data[x,:3] - self.subbox_data[y,:3])
                        dist_max = (np.linalg.norm(self.subbox_data[x,3:6]) +  np.linalg.norm(self.subbox_data[y,3:6])) / 2
                        if dist > dist_max: continue
                        self.subbox_edge_indices.append([x,y]) 
                        #self.subbox_edge_type.append(self.edge_type[:,i])
                        self.subbox_edge_type.append(self.subbox_data[x,:3] - self.subbox_data[y,:3])
        self.subbox_edge_indices = np.array(self.subbox_edge_indices).T
        self.subbox_edge_type = np.array(self.subbox_edge_type).T
        self.edge_index= np.array(self.edge_index).T
        self.edge_type = np.array(self.edge_type).T

        collisions,dist_after = self.getboxcollision()
        wall_collisions = self.getwallcollision()
     #   print(dist_after)

        print('regular' + str(collisions))
        print('wall' + str(wall_collisions))
        # base reward = -1
        robot_fraction = 0# (self.get_robot_viability() - 0.5) * 2
        print('---------------------------  ' + str(robot_fraction))
        #entr = self.get_entropy()
        reward +=  -collisions *1 + robot_fraction - wall_collisions * 50 + self.get_free_space()#+ (dist_after - dist_orig)# + #+ entr 
        #print( str(wall_collisions))
        #sizes = self.data[item,3] + self.data[item,4] + self.data[item,5]
        #order = np.argsort(sizes).argsort()[item]
        #reward -= sizes * 0.1
        if collisions < collisions_orig: reward += 10      #else: reward = -collisions 
        #reward = -collisions 
        # for i in range(self.data.shape[0]):
        #     if np.abs(self.data[i,0]) >= 2.0: reward += -10
        #     if np.abs(self.data[i,2]) >= 2.0: reward += -10

        #tmp = np.abs(self.data) > 8.0
        #reward += tmp[:,[0,2]].sum() * -5
        done = False
        # if collisions == 0 and wall_collisions == 0: 
        # #     reward = 150
        #     reward = 100
        #     done = True
        # #     self.done = True
        if wall_collisions_orig == 0 and wall_collisions != 0:
            done = True
        return float(reward),done

    def isdone(self):
        collisions,_ = self.getboxcollision()
        if collisions == 0: return True
        else: return False
    def visualize2D(self,save_path = None):
        plt.clf()
        fig = plt.figure("1")
        ax = fig.gca()
        ax.scatter(self.wall_countour[:,0],self.wall_countour[:,2],linewidths=0.1,marker='.')
        xmin = np.min(self.wall_countour[:,0])
        xmax = np.max(self.wall_countour[:,0])
        ymin = np.min(self.wall_countour[:,2])
        ymax = np.max(self.wall_countour[:,2])
        centers = []
        for object_index in range(np.size(self.data[:self.item_count_real],0)):

            object_descriptor = self.data[object_index]
            center = object_descriptor[0:3]
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

            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]

            dist = fastdist.matrix_to_matrix_distance((self.sample_points[object_index] + center)[:,[0,2]], self.wall_countour[:,[0,2]], fastdist.euclidean, "euclidean")
            category_str += ("_" + str(np.min(dist)))


            plt.text(minv[0],maxv[2],category_str)
            centerv = (minv + maxv) / 2
            centers.append(centerv)
        for object_index in range(self.subbox_data.shape[0]):
            object_descriptor = self.subbox_data[object_index]
            if object_descriptor[-1] != 1: continue
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]
         #   print(object_descriptor)

            translation = trimesh.transformations.translation_matrix(center)
            rotation = trimesh.transformations.quaternion_matrix(object_descriptor[6:10])
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            vert = b.vertices
            my_quaternion = Quaternion(object_descriptor[6:10])
            p1 = my_quaternion.rotate(np.array([extents[0] / 2,0,extents[2] / 2])) + center
            p2 = my_quaternion.rotate(np.array([extents[0] / 2,0,-extents[2] / 2])) + center
            p3 = my_quaternion.rotate(np.array([-extents[0] / 2,0,-extents[2] / 2])) + center
            p4 = my_quaternion.rotate(np.array([-extents[0] / 2,0,extents[2] / 2])) + center

          #  print(p1,p2,p3,p4,center)

            
            ax.add_line(Line2D([p1[0],p2[0]],[p1[2],p2[2]]))
            ax.add_line(Line2D([p2[0],p3[0]],[p2[2],p3[2]]))
            ax.add_line(Line2D([p3[0],p4[0]],[p3[2],p4[2]]))
            ax.add_line(Line2D([p4[0],p1[0]],[p4[2],p1[2]]))
        for index in range(len(self.box_collection)):
            object_descriptor = self.data[index]
            self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        index = 0
        ret = 0
        for i in range(len(self.mgrs)):
            for j in range(i+1, len(self.mgrs)):
                is_collision= self.mgrs[i].in_collision_other(self.mgrs[j])
                if is_collision:
                    ax.add_line(Line2D([self.data[i,0],self.data[j,0]],[self.data[i,2],self.data[j,2]]))
        plt.xlim((xmin - 0.8, xmax + 0.8))
        plt.ylim((ymin - 0.8, ymax + 0.8))
        canvas = FigureCanvasAgg(fig)

        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        return X
    def visualize2D_GUI(self,save_path = None,highlight_idx=0):
        plt.clf()
        fig = plt.figure("1")
        ax = fig.gca()
    #    wall_verts = self.wall[0]
        wall_coll = self.getwallcollision()
        if wall_coll > 0:
            ax.scatter(self.wall_countour[:,0],self.wall_countour[:,2],linewidths=0.1,marker='.',color='red')
        else:
            ax.scatter(self.wall_countour[:,0],self.wall_countour[:,2],linewidths=0.1,marker='.',color='blue')

        xmin = np.min(self.wall_countour[:,0])
        xmax = np.max(self.wall_countour[:,0])
        ymin = np.min(self.wall_countour[:,2])
        ymax = np.max(self.wall_countour[:,2])
        centers = []
        for object_index in range(np.size(self.data[:self.item_count_real],0)):

            object_descriptor = self.data[object_index]
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]

            translation = trimesh.transformations.translation_matrix(center)
            rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            minv = np.min(b.vertices,axis=0)
            maxv = np.max(b.vertices,axis=0)

            extents2 = b.bounding_box_oriented.primitive.extents
            transform2 = b.bounding_box_oriented.primitive.transform

            p1 = np.dot(rotation, np.array([extents[0] / 2,0,extents[2] / 2,1]))[0:3] + center
            p2 = np.dot(rotation, np.array([extents[0] / 2,0,-extents[2] / 2,1]))[0:3]+ center
            p3 = np.dot(rotation, np.array([-extents[0] / 2,0,-extents[2] / 2,1]))[0:3] + center
            p4 = np.dot(rotation, np.array([-extents[0] / 2,0,extents[2] / 2,1]))[0:3]  + center
            if(highlight_idx!=object_index):
                ax.add_line(Line2D([p1[0],p2[0]],[p1[2],p2[2]]))
                ax.add_line(Line2D([p2[0],p3[0]],[p2[2],p3[2]]))
                ax.add_line(Line2D([p3[0],p4[0]],[p3[2],p4[2]]))
                ax.add_line(Line2D([p4[0],p1[0]],[p4[2],p1[2]]))
            else:
                ax.add_line(Line2D([p1[0],p2[0]],[p1[2],p2[2]],color='red'))
                ax.add_line(Line2D([p2[0],p3[0]],[p2[2],p3[2]],color='red'))
                ax.add_line(Line2D([p3[0],p4[0]],[p3[2],p4[2]],color='red'))
                ax.add_line(Line2D([p4[0],p1[0]],[p4[2],p1[2]],color='red'))
            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)] + '(' + str(object_index) + ')'
            dist = fastdist.matrix_to_matrix_distance((self.sample_points[object_index] + center)[:,[0,2]], self.wall_countour[:,[0,2]], fastdist.euclidean, "euclidean")
            category_str += ("_" + str(np.min(dist)))
            plt.text(minv[0],maxv[2],category_str)
            centerv = (minv + maxv) / 2
            centers.append(centerv)
        for object_index in range(self.subbox_data.shape[0]):
            for i in range(len(self.tmp_subbox_length_sum) - 1):
                if object_index >= self.tmp_subbox_length_sum[i] and object_index < self.tmp_subbox_length_sum[i + 1]: break
            object_descriptor = self.subbox_data[object_index]
            if object_descriptor[-1] != 1: continue
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]

            translation = trimesh.transformations.translation_matrix(center)
            rotation = trimesh.transformations.quaternion_matrix(object_descriptor[6:10])
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            vert = b.vertices
            my_quaternion = Quaternion(object_descriptor[6:10])
            p1 = my_quaternion.rotate(np.array([extents[0] / 2,0,extents[2] / 2])) + center
            p2 = my_quaternion.rotate(np.array([extents[0] / 2,0,-extents[2] / 2])) + center
            p3 = my_quaternion.rotate(np.array([-extents[0] / 2,0,-extents[2] / 2])) + center
            p4 = my_quaternion.rotate(np.array([-extents[0] / 2,0,extents[2] / 2])) + center

          
            if(highlight_idx!=i):
                ax.add_line(Line2D([p1[0],p2[0]],[p1[2],p2[2]],color='green'))
                ax.add_line(Line2D([p2[0],p3[0]],[p2[2],p3[2]],color='green'))
                ax.add_line(Line2D([p3[0],p4[0]],[p3[2],p4[2]],color='green'))
                ax.add_line(Line2D([p4[0],p1[0]],[p4[2],p1[2]],color='green'))
            else:
                ax.add_line(Line2D([p1[0],p2[0]],[p1[2],p2[2]],color='red'))
                ax.add_line(Line2D([p2[0],p3[0]],[p2[2],p3[2]],color='red'))
                ax.add_line(Line2D([p3[0],p4[0]],[p3[2],p4[2]],color='red'))
                ax.add_line(Line2D([p4[0],p1[0]],[p4[2],p1[2]],color='red'))


        for index in range(len(self.box_collection)):
            object_descriptor = self.data[index]
            self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        index = 0
        ret = 0
        for i in range(len(self.mgrs)):
            for j in range(i+1, len(self.mgrs)):
                is_collision= self.mgrs[i].in_collision_other(self.mgrs[j])
                if is_collision:
                    ax.add_line(Line2D([self.data[i,0],self.data[j,0]],[self.data[i,2],self.data[j,2]]))
        plt.xlim((xmin - 0.8, xmax + 0.8))
        plt.ylim((ymin - 0.8, ymax + 0.8))
        canvas = FigureCanvasAgg(fig)

        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        return X
    def getwallcollision(self):
        for index in range(len(self.box_collection)):
            object_descriptor = self.data[index]
            self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        index = 0
        for i in range(len(self.mgrs)):
            is_collision= self.mgrs[i].in_collision_other(self.wall_collider)
            if is_collision:
                index += 1
        return index

    def getwallcollisiondetail(self):
        for index in range(len(self.box_collection)):
            object_descriptor = self.data[index]
            self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        index = 0
        for i in range(len(self.mgrs)):
            is_collision= self.mgrs[i].in_collision_other(self.wall_collider)
            if is_collision:
                return i
        
    
# return number of collision
    def getboxcollision(self):
        for index in range(len(self.box_collection)):
            object_descriptor = self.data[index]
            self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        index = 0
        ret = 0
        collision_dist = {}
        for i in range(len(self.mgrs)):
            for j in range(i+1, len(self.mgrs)):
                is_collision,l_contact= self.mgrs[i].in_collision_other(self.mgrs[j],return_data = True)
                if is_collision:
                    ret += 1#len(l_contact) * 0.05
                    collision_dist[(i,j)] = np.linalg.norm(self.data[i,:3] - self.data[j,:3])
                    index += 1
       # self.edge_index = np.array(edge_idx_tmp).T
       # self.edge_type = np.eye(3)[0].reshape(1,3).repeat(self.edge_index.shape[1],axis=0).T
        #print(self.edge_index)
        return ret,collision_dist

    def getboxcollisionpair(self):
        ret= []
        for index in range(len(self.box_collection)):
            object_descriptor = self.data[index]
            self.mgrs[index].set_transform(' ',trimesh.transformations.translation_matrix(object_descriptor[0:3]))
        index = 0
        for i in range(len(self.mgrs)):
            for j in range(i+1, len(self.mgrs)):
                is_collision,l_contact= self.mgrs[i].in_collision_other(self.mgrs[j],return_data = True)
                if is_collision:
                    ret.append((i,j))
       # self.edge_index = np.array(edge_idx_tmp).T
       # self.edge_type = np.eye(3)[0].reshape(1,3).repeat(self.edge_index.shape[1],axis=0).T
        #print(self.edge_index)
        return ret

    def get_robot_viability(self):
        #print(self.OutputMoveitSceneFile())
        return do_planning(self.OutputMoveitSceneFile())


    def get_entropy(self):
        vertices = list()
        codes = [Path.MOVETO]
        obj_count = 0
        centers = []
        for object_index in range(self.item_count_real):
            
            object_descriptor = self.data[object_index]
            category_onehot = list(object_descriptor[7:])
            if category_class[category_onehot.index(1)] == 'Lamp':continue

            codes += [Path.MOVETO] + [Path.LINETO]*(3) + [Path.CLOSEPOLY]
            vertices.append((0,0)) #Always ignored by closepoly command
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]

            translation = trimesh.transformations.translation_matrix(center)
            rotation = rotation_matrix(np.array([0,1,0]),object_descriptor[6])
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            minv = np.min(b.vertices,axis=0)
            maxv = np.max(b.vertices,axis=0)

            vertices.append((minv[0],minv[2]))
            vertices.append((maxv[0],minv[2]))
            vertices.append((maxv[0],maxv[2]))
            vertices.append((minv[0],maxv[2]))
            centers.append((object_descriptor[0],object_descriptor[2]))
            obj_count += 1

                    
        vertices.append((0,0))
        #print(vertices)
        vertices = np.array(vertices, float)
        obstacles = Path(vertices, codes)

        
        #plotter    = ImageGenerator()
        #plotter.draw_obstacle_course(obstacles)
        
        KDs = []
        idx = 0
        while True:
            for i in range(obj_count):
                for j in range(i + 1, obj_count):
                    start = centers[i]
                    goal = centers[j]
                    kd = RRT(obstacles, start, goal, 0.1, 200,i,j)
                    print(idx)
                    idx=idx+1
                    if kd is not None:
                        KDs+=kd
                    if idx == 100: break
                if idx == 100: break
            if idx == 100: break
        tree = kdtree.create(KDs)
       

       # tree.printTree(tree.root)

        minp, maxp   = obstacles.vertices.min(0), obstacles.vertices.max(0)
        width = maxp[0] - minp[0] + 1
        height = maxp[1] - minp[1] + 1
        obs = Obstacles(obstacles.to_polygons())
        cmap = plt.get_cmap('plasma')
        x = np.zeros((64,64))
        c = np.zeros((64,64,3),dtype='uint8')
        s = 0
        for i in range(64):
            for j in range(64):
                point = [width * i / 64 - width / 2, height * j / 64 - height / 2]
                if not obs.point_is_valid(point[0],point[1]):
                    x[j,i] = -1
                    continue
                #print(point)
                x[j,i] = len(tree.search_nn_dist(point, 0.05))
                s += x[j,i]
                #print(x[i,j])
        
        x /= s
        return entr(x).sum() / np.log(2)
        # x /= np.max(x)
        # for i in range(64):
        #     for j in range(64):
        #         v = x[i,j]
        #         if v < 0 :
        #             c[i,j,:] = np.array([0.5,0.5,0.5]) * 255
        #             continue
        #         ctemp = cmap(v)
        #         c[i,j,:] = np.array([ctemp[2],ctemp[1],ctemp[0]]) * 255
        # cv2.imshow('image',c)

        # cv2.waitKey(0) 

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
    def output_json_2(self,path:str):
        fpp = self.file_path
        with open(fpp) as f:
            json_data = json.load(f)
        for i in range(len(json_data['objects'])):
            json_data['objects'][i]['box'][1:4] = self.data[i,[6,5,4]]

        with open(path,'w') as f:
            json.dump(json_data,f)
    def output_render_result(self,path:str):
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        renderer = pyrender.OffscreenRenderer(512, 512)

        scene = pyrender.Scene()
        scene_trimesh = trimesh.Scene()
        with open(self.file_path) as f:
            json_data = json.load(f)

        floor_mesh = trimesh.creation.extrude_polygon(self.wall_contour_poly,1)
        
        floor_mesh2=trimesh.Trimesh(vertices=floor_mesh.vertices[:,[0,2,1]],faces=floor_mesh.faces)
        trans = trimesh.transformations.translation_matrix([0,-0.5 + self.floor_level,0])
        floor_mesh2.apply_transform(trans)
        fmesh = pyrender.Mesh.from_trimesh(floor_mesh2)
        scene.add(fmesh)
        for i in range(self.item_count_real):
            partnetdata = '/home/yangjie/202/data/data_v0'
            processed_partnetdata = '/home/yangjie/202/data/structurenet_hire'
            obj = json_data['objects'][i]
            box_para = np.array(obj['box'])
            


            partnet_id = obj['model_id'].split('/')[0]
            partnet_cat = json.load(open(os.path.join(partnetdata, partnet_id, 'meta.json')))['model_cat']
            objfile = os.path.join(processed_partnetdata, partnet_cat, partnet_id, 'objs/0.obj')
            mesh = trimesh.load(objfile,force='mesh')
            extent = mesh.bounding_box.extents
            scale = np.zeros((4,4))
            scale[0,0] = box_para[3] / extent[0]
            scale[1,1] = box_para[2] / extent[1]
            scale[2,2] = box_para[1] / extent[2]
            scale[3,3] = 1
            r1 = trimesh.transformations.euler_matrix(0, box_para[0],0)
            r2 =trimesh.transformations.euler_matrix(0, 3.1415926/2,0)
            t1 =  trimesh.transformations.translation_matrix(box_para[[4,5,6]])
            matrix = np.dot(np.dot(np.dot(t1,r1),r2),scale)
            mesh.apply_transform(matrix)

            scene_trimesh.add_geometry(mesh)
            mesh2 = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh2)
        scene_trimesh.add_geometry(floor_mesh2)
        #scene_trimesh.show()
        trans = (scene_trimesh.bounds[0] + scene_trimesh.bounds[1]) / 2
        max_ext = np.max(np.abs(scene_trimesh.bounds[0] - scene_trimesh.bounds[1]))
        camera = pyrender.OrthographicCamera(xmag=max_ext / 2,ymag=max_ext / 2)
        camera_pose = np.array([
        [-1, 0, 0, trans[0]],
        [0 , 0, 1, 5.],
        [0 , 1, 0, trans[2]],
        [0 , 0, 0, 1         ],
        ])
        #x[f[:-4]] = np.array([max_ext / max_ext2, trans[0,3], trans[2,3]])
        scene.add(camera, pose=camera_pose)
        color, depth = renderer.render(scene)
        #cv.imshow('a',color)
        #cv.waitKey(0)
        p = depth[np.where(depth!= 0)]
        pmin = np.min(p)
        pmax = np.max(p)
        depth[np.where(depth!= 0)] -= (pmin - 0.0001)
        depth[np.where(depth!= 0)] /= ((pmax-pmin) * 2)
        depth[np.where(depth!= 0)] += 0.5
        cv.imwrite(path,depth*255)



    def OutputMoveitSceneFile(self):
       # with open(filename,'w') as f:
        output = []
        boxes = self.subbox_data
        scene_tmp = trimesh.Scene()
        # for box in boxes:
        #     if box[-1] == 1: continue
        #     box = box.flatten()
        #     center = box[0:3]
        #     extents = box[3:6]
        #     rot = box[6:10]
            
        #     translation = trimesh.transformations.translation_matrix(center)
        #     rotation = trimesh.transformations.quaternion_matrix(rot)
        #     b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
        #     #b.visual.vertex_colors = (255 * np.array(visualize_color[category_str])).astype(np.uint8)
        #     scene_tmp.add_geometry(b)
        # #scene_tmp.rezero()
        # bounds = (scene_tmp.bounds)
        # tmp_extents = bounds[1,:] - bounds[0,:]
        # tmp_center = (bounds[1,:] + bounds[0,:] ) / 2

        i = 0
        idx = 0
        for box in boxes:
            i = i + 1
            if i > self.subbox_lengths[idx]:
                i = 0
                idx += 1
            if box[-1] == 1:continue
            box = box.flatten()
            center = box[0:3]
            extents = box[3:6]
            rot = box[6:10]
            #center = center - tmp_center
            #center = center * (object_descriptor[3:6]) / (tmp_extents)
            #extents = extents * (object_descriptor[3:6]) / (tmp_extents)
            #center = np.dot(rotation_matrix(np.array([0,1,0]),object_descriptor[6])[0:3,0:3],center)
            #center = center + object_descriptor[0:3]
            center = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675)[:3,:3],center)
            
            matrix = trimesh.transformations.quaternion_matrix(rot)
            matrix = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675),matrix)
            rot = trimesh.transformations.quaternion_from_matrix(matrix)
            translation = trimesh.transformations.translation_matrix(center)
            rotation = trimesh.transformations.quaternion_matrix(rot)
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,matrix))
            scene_tmp.add_geometry(b)
            output.append('box ' + str(center[0]) + ' ' +  str(center[1]) + ' ' +str(center[2]) + ' ' + str(extents[0]) + ' ' + str(extents[1]) + ' ' +str(extents[2]) + ' ' + str(rot[0]) + ' ' +str(rot[1]) + ' ' +str(rot[2]) +' ' + str(rot[3]) + ' ' + str(idx))
        for object_index in range(np.size(self.data,0)):
            object_descriptor = self.data[object_index]
            category_onehot = list(object_descriptor[7:])
            category_str = category_class[category_onehot.index(1)]
            # print(category_str)
            if(category_str.find('Moveable') == -1):
                continue                   
            else:
                for (k,v) in self.moveable_dict.items():
                    if object_index in v:
                        belong_idx = k
                # print('123123123')
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
                    center = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675)[:3,:3],center)
                    translation = trimesh.transformations.translation_matrix(center)
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(matrix,translation_1)))
                    b.visual.vertex_colors = (255 * np.array([0.3,0.3,0.3,0.3])).astype(np.uint8)
                    scene_tmp.add_geometry(b)
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,np.dot(matrix,translation_2)))
                    b.visual.vertex_colors = (255 * np.array([0.3,0.3,0.3,0.3])).astype(np.uint8)
                    scene_tmp.add_geometry(b)
                    output.append('moveable ' + str(center[0]) + ' ' +  str(center[1]) + ' ' +str(center[2]) + ' ' + str(extents[0]) + ' ' + str(extents[1]) + ' ' +str(extents[2]) + ' ' + str(rot[0]) + ' ' +str(rot[1]) + ' ' +str(rot[2]) +' ' + str(rot[3]) + ' ' + str(belong_idx))

                    # get the slide line origin in local frame
                    slide_orig = slide_direction * (extents - object_descriptor[17])
                    slide_dest = slide_orig + slide_direction * object_descriptor[18]
                    slide_orig = np.dot(matrix[:3,:3], slide_orig) + center
                    slide_dest = np.dot(matrix[:3,:3], slide_dest) + center

                    output.append('path line ' + str(slide_orig[0]) + ' ' +  str(slide_orig[1]) + ' ' +str(slide_orig[2]) + ' ' + str(slide_dest[0]) + ' ' + str(slide_dest[1]) + ' ' +str(slide_dest[2]))
                    
                else:
                    rotation_orig = object_descriptor[14:17]
                    rotation_direction = object_descriptor[17:20]
                    # rotation_orig = np.dot(rotation_orig,rotation[0:3,0:3])
                    #rotation_orig = rotation_orig / np.linalg.norm(rotation_orig)
                    
                    rotation_1 =  trimesh.transformations.rotation_matrix(object_descriptor[21], rotation_direction,rotation_orig)
                    rotation_2 =  trimesh.transformations.rotation_matrix(object_descriptor[20], rotation_direction,rotation_orig)

                    matrix = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675),(rotation_matrix(np.array([0,1,0]),object_descriptor[6])))
                    rot = trimesh.transformations.quaternion_from_matrix(matrix)
                    translation = trimesh.transformations.translation_matrix(center)
                    center = np.dot(rotation_matrix(np.array([1,0,0]),1.57079632675)[:3,:3],center)
                    translation__ = trimesh.transformations.translation_matrix(center)
                    rotation = trimesh.transformations.quaternion_matrix(rot)
                    
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation__,np.dot(matrix,rotation_1)))
                    b.visual.vertex_colors = (255 * np.array([0.3,0.3,0.3,0.3])).astype(np.uint8)
                    scene_tmp.add_geometry(b)
                    b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation__,np.dot(matrix,rotation_2)))
                    b.visual.vertex_colors = (255 * np.array([0.3,0.3,0.3,0.3])).astype(np.uint8)
                    scene_tmp.add_geometry(b)

                    output.append('moveable ' + str(center[0]) + ' ' +  str(center[1]) + ' ' +str(center[2]) + ' ' + str(extents[0]) + ' ' + str(extents[1]) + ' ' +str(extents[2]) + ' ' + str(rot[0]) + ' ' +str(rot[1]) + ' ' +str(rot[2]) +' ' + str(rot[3]) + ' ' + str(belong_idx))

                    rotdir_main_idx = np.argmax(np.abs(rotation_direction ))
                    extent_min_idx = np.argmin(extents)
                    extents_tmp = extents
                    rotation_orig[[rotdir_main_idx,extent_min_idx]] = [0,0]
                    radius = 2* np.linalg.norm(rotation_orig)
                    rotation_center = rotation_orig
                    rotation_orig = -rotation_orig

                    rotation_direction = np.dot(matrix[:3,:3], rotation_direction)
                    rotation_center = np.dot(matrix[:3,:3], rotation_center) + center
                    rotation_orig = np.dot(matrix[:3,:3], rotation_orig) + center

                    

                    output.append('path arc ' + str(rotation_center[0]) + ' ' +  str(rotation_center[1]) + ' ' + str(rotation_center[2]) + ' ' + str(rotation_orig[0]) + ' ' +  str(rotation_orig[1]) + ' ' +str(rotation_orig[2]) + ' ' + str(radius) + ' ' +  str(object_descriptor[21] - object_descriptor[20]) + ' ' + str(rotation_direction[0]) + ' ' +str(rotation_direction[1]) + ' ' +str(rotation_direction[2]))
        return output
               #scene_tmp.show()
    #exit(0)