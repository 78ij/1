


import pyrender

import trimesh
import os
import cv2 as cv
import json
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
#flags = pyrender.SKIP_CULL_FACES
renderer = pyrender.OffscreenRenderer(512, 512)



overall_path = '/home/yangjie/208/data4T/yangjie/Sync2Gen/optimization/render_scenes/bedroom/'

scene_lst = os.listdir(overall_path)
for folder in scene_lst:
    pp = overall_path + '/' + folder
    lst2 = os.listdir(pp)
    scene = pyrender.Scene()
    scene_trimesh = trimesh.Scene()
    scene_furniture = pyrender.Scene()
    scene_floor = pyrender.Scene()
    for ff in lst2:
        if(ff.find('.obj') == -1):
            continue
        

        mesh = trimesh.load(overall_path + '/' + folder + '/' + ff,force='mesh')
       #mesh.show()
        mesh.visual= trimesh.visual.ColorVisuals()
        scene_trimesh.add_geometry(mesh)
        mesh2 = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh2)
        if(ff.find('000') != -1):
            scene_floor.add(mesh2)
        else:
            scene_furniture.add(mesh2)
    #scene_trimesh.show()
    #scene_trimesh.show()
   # print((scene_trimesh.bounds[0] + scene_trimesh.bounds[1]) / 2)
    #print(np.abs(scene_trimesh.bounds[0] - scene_trimesh.bounds[1]))
    ext = np.abs(scene_trimesh.bounds[0] - scene_trimesh.bounds[1])
    trans = (scene_trimesh.bounds[0] + scene_trimesh.bounds[1]) / 2
    max_ext = np.max(np.abs(scene_trimesh.bounds[0] - scene_trimesh.bounds[1]))
    camera = pyrender.OrthographicCamera(xmag=max_ext / 2,ymag=max_ext / 2)
    camera_pose = np.array([
    [-1, 0, 0, trans[0]],
    [0 , 0, 1, trans[1] + ext[1]],
    [0 , 1, 0, trans[2]],
    [0 , 0, 0, 1         ],
    ])
    #x[f[:-4]] = np.array([max_ext / max_ext2, trans[0,3], trans[2,3]])
    scene.add(camera, pose=camera_pose)
    
    color, depth1 = renderer.render(scene)
    #cv.imshow('a',color)
    #cv.waitKey(0)
    p = depth1[np.where(depth1!= 0)]
    pmin = np.min(p)
    pmax = np.max(p)
    depth1[np.where(depth1!= 0)] -= (pmin - 0.0001)
    depth1[np.where(depth1!= 0)] /= ((pmax-pmin) * 2)
    depth1[np.where(depth1!= 0)] += 0.5

    scene_floor.add(camera, pose=camera_pose)
    color, depth2 = renderer.render(scene_floor)
    p = depth2[np.where(depth2!= 0)]
    pmin = np.min(p)
    pmax = np.max(p)
    print(pmin)
    print(pmax)
    depth2[np.where(depth2!= 0)] = 1

    scene_furniture.add(camera, pose=camera_pose)
    color, depth3 = renderer.render(scene_furniture)
    p = depth3[np.where(depth3!= 0)]
    pmin = np.min(p)
    pmax = np.max(p)
    depth3[np.where(depth3!= 0)] = 1

    dd=np.zeros((512,512,3))
   # dd[:,:,0] = depth1
    dd[:,:,1] = depth2
    dd[:,:,2] = depth3
    
    cv.imwrite('./rendered/train/bad/sync_bedroom_' + folder + '.png',dd*255)
  #  exit(1)