import os
import trimesh
from common import *
import numpy as np
lst_obj = os.listdir(FUTURE_PATH + '/models/')

idx = 0
for obj in lst_obj:
    mesh = trimesh.load(FUTURE_PATH + '/models/' + obj + '/raw_model.obj',force='mesh',process=False)
    mmin = np.min(mesh.vertices,axis=0) 
    mmax = np.max(mesh.vertices,axis=0)
    print(str(idx) + '/' + str(len(lst_obj)))
    idx += 1
    with open(FUTURE_PATH + '/models/' + obj + '/bbox.txt','w') as f:
        f.write(str(mmax[0]) + ' ' + str(mmax[1]) + ' ' + str(mmax[2]) + ' ' + str(mmin[0]) + ' ' + str(mmin[1]) + ' ' + str(mmin[2]))