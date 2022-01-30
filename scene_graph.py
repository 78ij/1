import json
import os
import numpy as np
import torch
from common import *
import trimesh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx
import pickle
from utils import load_pts, transform_pc
import chamfer_distance
import torch


unit_cube = load_pts('cube.pts')
def min_dist(data_1,data_2):
    pc_1 = transform_pc(torch.tensor(unit_cube), torch.tensor(data_1)).float()
    pc_2 = transform_pc(torch.tensor(unit_cube), torch.tensor(data_2)).float()
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


class BasicSceneGraph:
    def __init__(self, data: np.array, edge_index: np.array,edge_type: np.array,room_type:str):
        self._data = data
        self._edge_index = edge_index
        self._room_type = room_type
        self._edge_type = edge_type

    def visualize2D(self):
        fig = plt.figure("2D Top-Down")
        ax = fig.add_subplot(111, aspect='equal')
        G = networkx.DiGraph()
        centers = []
        for object_index in range(np.size(self._data,0)):

            object_descriptor = self._data[object_index]
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]
            rot = object_descriptor[6:10]

            translation = trimesh.transformations.translation_matrix(center)
            rotation = trimesh.transformations.quaternion_matrix(rot)
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            minv = np.min(b.vertices,axis=0)
            maxv = np.max(b.vertices,axis=0)
    
            ax.add_line(Line2D([minv[0],maxv[0]],[minv[2],minv[2]]))
            ax.add_line(Line2D([maxv[0],maxv[0]],[minv[2],maxv[2]]))
            ax.add_line(Line2D([maxv[0],minv[0]],[maxv[2],maxv[2]]))
            ax.add_line(Line2D([minv[0],minv[0]],[maxv[2],minv[2]]))

            centerv = (minv + maxv) / 2
            centers.append(centerv)

            category_onehot = list(object_descriptor[10:])
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
        plt.show()     
        pos = networkx.spectral_layout(G)
        node_labels = networkx.get_node_attributes(G, 'label')
        networkx.draw_networkx_labels(G, pos, labels=node_labels)
        networkx.draw(G)
        plt.show()

    def visualize(self):
        scene = trimesh.Scene()
        G = networkx.DiGraph()
        for object_index in range(np.size(self._data,0)):

            object_descriptor = self._data[object_index]
            center = object_descriptor[0:3]
            extents = object_descriptor[3:6]
            rot = object_descriptor[6:10]

            translation = trimesh.transformations.translation_matrix(center)
            rotation = trimesh.transformations.quaternion_matrix(rot)
            b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
            scene.add_geometry(b)

            category_onehot = list(object_descriptor[10:])
            category_str = category_class[category_onehot.index(1)]
            G.add_node(object_index,label = category_str)
        if self._edge_index.size != 0:
            for edge_index in range(np.size(self._edge_index,1)):
                G.add_edge(self._edge_index[0,edge_index], self._edge_index[1,edge_index])


        scene.show()

        pos = networkx.random_layout(G)
        node_labels = networkx.get_node_attributes(G, 'label')
        networkx.draw_networkx_labels(G, pos, labels=node_labels)
        networkx.draw(G)
        plt.show()

    @staticmethod
    def getGraphsfromFRONTFile(path:str):
        # cope with 3D boundingbox now
        with open(path) as f:
            json_data = json.load(f)
        furniture_list = json_data['furniture']
        room_list = json_data['scene']['room']
        
        with open(FUTURE_PATH + '\\model_info.json') as f:
            model_category_data_raw = json.load(f)
        model_category_data = {}
        for model in model_category_data_raw:
            model_category_data[model['model_id']] = model['category']
        
        ret = []

        for room in room_list:
            data = []
            normalize_scene = trimesh.Scene()
            furniture_category_list = []
            for object_in_room in room['children']:
                for furniture in furniture_list:
                    if furniture['uid'] == object_in_room['ref'] and furniture.__contains__('valid') and furniture['valid']:
                        furniture_id = furniture['jid']
                        pos = np.array(object_in_room['pos'])
                        rot =  np.array(object_in_room['rot'])
                        scale =  np.array(object_in_room['scale'])
                        if not os.path.exists(FUTURE_PATH + '\\3D-FUTURE-model\\' + furniture_id + '\\raw_model.obj'):
                            continue
                        #furniture_mesh = trimesh.load(FUTURE_PATH + '\\3D-FUTURE-model\\' + furniture_id + '\\raw_model.obj',force='mesh',process=False)

                        #vert = furniture_mesh.vertices.copy()
                        #vert = vert.astype(np.float64) * scale

                        with open(FUTURE_PATH + '\\3D-FUTURE-model\\' + furniture_id + '\\bbox.txt') as f:
                            line = f.readline().split()
                            line = [float(number) for number in line]
                            maxv = np.array(line[0:3])
                            minv = np.array(line[3:6])

                       # print(minv)
                        minv = minv * scale
                        maxv = maxv * scale
                        maxv += pos
                        minv += pos
                        # ref = [0,0,1]
                        # axis = np.cross(ref, rot[1:])
                        # theta = np.arccos(np.dot(ref, rot[1:]))*2
                        # if axis[1] < 0: theta = -theta

                        center = (maxv + minv) / 2
                        extent = (maxv - minv)

                        translation = trimesh.transformations.translation_matrix(center)
                        rotation = trimesh.transformations.quaternion_matrix(rot)

                        box = trimesh.primitives.Box(extents = maxv - minv,
                        transform = np.dot(translation,rotation))
                        normalize_scene.add_geometry(box)
                        furniture_category = categories[model_category_data[furniture_id]]
                        
                        cate = np.eye(len(category_class))[category_class.index(furniture_category)]
                        furniture_category_list.append(cate)

            if len(furniture_category_list) == 0: continue
            scale_factor = np.max(normalize_scene.bounding_box.extents)
            #print(normalize_scene.bounding_box.extents)
            normalize_scene = normalize_scene.scaled(1 / scale_factor)
            scene_center = (normalize_scene.bounds[0] + normalize_scene.bounds[1]) / 2
            # 3 + 3 + 1 + 5 = 12
            objs = normalize_scene.dump()
            for i in range(len(objs)):
                box_i = objs[i]
                extent = box_i.primitive.extents / scale_factor
                center = (np.max(box_i.vertices,axis=0) + np.min(box_i.vertices,axis=0)) / 2 - scene_center
                transform = box_i.primitive.transform
                scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(transform)
                rot = trimesh.transformations.quaternion_from_euler(angles[0],angles[1],angles[2])
                cate = furniture_category_list[i]
                data.append(np.concatenate([center,extent,rot,cate]))
            # Basic: A Dense Graph of the edges
            #edge_index = make_dense_edge_index(len(data))
            # Level1 : Graph of Nearby Objects and same Objects
            # type 0: Normal spatial
            # type 1: same
            edge_index = []
            edge_type = []
            for i in range(len(data)):
                for j in range(i, len(data)):
                    if min_dist(data[i][:10],data[j][:10]) < 0.01 and i != j:
                        edge_index.append([i,j])
                        edge_index.append([j,i])
                        edge_type.append(0)
                        edge_type.append(0)
                    if np.linalg.norm(data[i][3:6] - data[j][3:6]) < 0.001 and i != j:
                        edge_index.append([i,j])
                        edge_index.append([j,i])
                        edge_type.append(1)
                        edge_type.append(1)
                        
            data = np.array(data)
            #print(edge_index)
            #print(edge_type)

            edge_index = np.array(edge_index).T
            edge_type = np.array(edge_type).T
            ret.append(BasicSceneGraph(data,edge_index,edge_type,room['type']))

        return ret
        
if __name__=="__main__":

    graphs = BasicSceneGraph.getGraphsfromFRONTFile('H:\\GRAINS-master\\0-data\\3D-FRONT\\00b88e19-d106-4ab8-a322-31c494a0a6b9.json')
    for graph in graphs:
        graph.visualize2D()

    # with open('complete_data.pkl','rb') as f:
    #     loaded = pickle.load(f)
    # graph = BasicSceneGraph(loaded['data'][11166],loaded['edge_index'][11166],'type')
    # graph.visualize()