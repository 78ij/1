import numpy as np
import pickle
from common import *

with open('complete_data_structurenet.pkl', 'rb') as f:
    orig_data = pickle.load(f)

processed_data = []

k = 0
for data in orig_data:
    k = k + 1
    print(k)
    data_tmp = np.zeros([data._data.shape[0],data._data.shape[1] + 3 + 3 + 2])
    for i in range(data._data.shape[0]):
        # 3 cent 3 extent 1 angle 7 cate 128 rootc 8 moveable
        data_tmp[i,0:142] = data._data[i,:]
        category_onehot = list(data_tmp[i,7:])
        category_str = category_class[category_onehot.index(1)]
        if category_str == 'Moveable_Slider':
            data_tmp[i,145:150] = data._data[i,14:19]
            data_tmp[i,14:142] = np.zeros(128)
        elif category_str == 'Moveable_Revolute':
            data_tmp[i,142:150] = data._data[i,14:22]
            data_tmp[i,149:150] /= 3.1415926535
            data_tmp[i,14:142] = np.zeros(128)
    data._data = data_tmp
    processed_data.append(data)

with open('complete_data_modified.pkl','wb') as f:
    pickle.dump(processed_data,f)