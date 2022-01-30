import torch
from torch.utils.data import DataLoader,Dataset
import os
from common import *
import numpy as np
import scene_graph_structurenet
import pickle

class SceneGraphDataSet(Dataset):
    def __init__(self, preload_path = None, filter_type = None,valid = False):
        if preload_path != None:
            with open(preload_path, 'rb') as f:
                self._data = pickle.load(f)      
        else:
            self._data = SceneGraphDataSet.process_data(filter_type = filter_type)
    

        # hardcoded: filter out all type mismatch scene
        # and scene that has object num > 20
        self._data = [data for data in self._data if np.size(data._data,0) <= 20 and data._room_type == filter_type]
        self.valid = valid

    def __getitem__(self, index):
        if not self.valid:
            #print(index)
            return (self._data[index]._data, self._data[index]._edge_index, self._data[index]._edge_type)
        else:
            #print(len(self._data) - 500 + index)
            return (self._data[len(self._data) - 500 + index]._data, self._data[len(self._data) - 500 + index]._edge_index, self._data[len(self._data) - 500 + index]._edge_type)

    def __len__(self):
        if not self.valid:
            #return len(self._data) - 500
            return 2000
        else:
            #return 1
            return 500
    @staticmethod
    def process_data(filter_type = None, save_path = None):
        candidate_graphs = []
        
        max_node = 0
        front_json_list = os.listdir(FRONT_PATH)
      #  with open(save_path,'rb') as f:
      #      candidate_graphs = pickle.load(f)
        num_processed = 1
        for front_json in front_json_list:
            
            print('Processing Data: ' + str(num_processed) + '/' + str(len(front_json_list)))
            num_processed += 1
            #if num_processed<= 2370: continue
            candidate_graphs += scene_graph_structurenet.BasicSceneGraph.getGraphsfromFRONTFile(FRONT_PATH + '/' + front_json)
            if num_processed % 100 == 0:
                if filter_type is not None:
                    candidate_graphs = [graph for graph in candidate_graphs if graph._room_type == filter_type]
                if save_path is not None:
                    with open(save_path,'wb') as f:
                        pickle.dump(candidate_graphs,f)
        if filter_type is not None:
            candidate_graphs = [graph for graph in candidate_graphs if graph._room_type == filter_type]

        if save_path is not None:
            with open(save_path,'wb') as f:
                pickle.dump(candidate_graphs,f)
        
        return candidate_graphs

if __name__ == '__main__':
    #SceneGraphDataSet.process_data(save_path = './complete_data_subbox_structurenet_withoutlamp.pkl')
   # with open('./complete_data_subbox_structurenet_withoutlamp.pkl','rb') as f:
    #    loaded = pickle.load(f)
    
    # i = 0
    #for data in loaded:
    #    if (np.size(data._data,0)) > 20:
    #         i += 1
    # print(i)

    dataset = SceneGraphDataSet('./complete_data_subbox_structurenet_withoutlamp.pkl','LivingDiningRoom',valid=False)
    #dataset._data[0].visualize()
    #for i in range(500):
        #print(i)
        #dataset._data[i].visualize2D('./vised_living/' + str(i) + '.png')
        #dataset._data[i].visualize()
    # tmpdata = []
    # ds = [0,4,6,7,10,11,13,15,18,25]
    # for i in (ds):
    #     tmpdata.append(dataset._data[i])
    #     print(dataset._data[i]._data[:,0:20])
    #    dataset._data[i].visualize()
  #  #    print(i)
    with open('./GraphAdjust/RL_train_data_ldroom_complete_subbox_2.pkl','wb') as f:
        pickle.dump(dataset._data,f)