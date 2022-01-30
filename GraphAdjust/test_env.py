import env_subgraph
import trimesh
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scene_graph_structurenet import BasicSceneGraph

with open('baseline_sync_bedroom_render.pkl','rb') as f:
    train_data = pickle.load(f)
train_data_size = len(train_data)

env_list = []
for i in range(train_data_size):
    print(i)
    try:
        env_tmp = env_subgraph.ENV(train_data[i])
       # if env_tmp.getboxcollision() != 0:
        env_list.append(env_tmp)
        print('xxx')
        if len(env_list) == 100: break
    except:
        continue
    

idx = 0
for env_tmp in env_list:
    #for x in range(10):
    env_tmp.output_render_result('./rendered/bad/sync_bedroom_' + str(idx) + '.png')
    idx += 1
    print(idx)
    # scene = trimesh.Scene()
    # #X = env_tmp.visualize2D()
    # env_tmp.get_free_space()
    # X = env_tmp.visualize2D()
    
    # plt.imshow(X)
    # plt.show()

    # for box in env_tmp.subbox_data:
    #     center = box[0:3]
    #     extents = box[3:6]
    #     rot = box[6:10]
    #     translation = trimesh.transformations.translation_matrix(center)
    #     rotation = trimesh.transformations.quaternion_matrix(rot)
    #     b = trimesh.primitives.Box(extents=extents.reshape(3,),transform=np.dot(translation,rotation))
    #     b.visual.vertex_colors = (255 * np.array([0.3,0.3,0.3,0.3])).astype(np.uint8)
    #     scene.add_geometry(b)
        
    # scene.show()
    # print(env_tmp.subbox_edge_indices)
    # print(env_tmp.edge_index)
    # for i in range(2,env_tmp.subbox_edge_indices.T.shape[0]):
    #     #print(env_tmp.subbox_edge_indices.T[i])
    #     c1 = env_tmp.subbox_data[env_tmp.subbox_edge_indices.T[i,0],0:3]
    #     c2 = env_tmp.subbox_data[env_tmp.subbox_edge_indices.T[i,1],0:3]
    #     #print(env_tmp.subbox_edge_indices.T[i])
    #     #print(c1)
    #     #print(c2)
    #     p = trimesh.load_path([c1,c2])
    #     scene.add_geometry(p)
    # scene.show()
    #scene.show()
        #env_tmp.step(0)
        #env_tmp.get_entropy()
    
   #env_tmp.OutputMoveitSceneFile('aaa')