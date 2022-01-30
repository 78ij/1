import trimesh
import json
import os
import numpy as np
import math


def traverse_part_tree(tree):
    dict_tmp = {'id':tree['id']}
    if tree.__contains__('objs'):
        dict_tmp['objs'] = tree['objs']
    lst = [dict_tmp]

    if tree.__contains__('children'):
        for child in tree['children']:
            lst += traverse_part_tree(child)
    return lst
def get_moveable_box_data(model_id):
    boxes_ret = []
    i = 0
    #valid_item_lst = ['Door','StorageFurniture','Chair','Table','Window']
    #valid_item_lst = ['StorageFurniture']
    #for folder in os.listdir('B:/partnet-mobility-v0_2/dataset'):
    i = i + 1
    #print(i)

    with open('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/meta.json') as f:
        json_data = json.load(f)


    with open('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/result.json') as f:
        segmentation_data = json.load(f)

    with open('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/mobility_v2.json') as f:
        mobility_data = json.load(f)
    scene = trimesh.Scene()
    for obj in os.listdir('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/textured_objs'):
        if obj.find('.obj') == -1: continue
        scene.add_geometry(trimesh.load('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/textured_objs/' + obj,process=False))

    orig_scene_extents = scene.bounding_box.primitive.extents
    orig_scene_center = (scene.bounds_corners.min(axis=0) + scene.bounds_corners.max(axis=0)) / 2
    all_part_list = traverse_part_tree(segmentation_data[0])

    for mobility_part in mobility_data:
        mobility_meshs = []
        for parts in  mobility_part['parts']:
            part_id = parts['id']
            for part in all_part_list:
                if part['id'] == part_id:
                    mobility_meshs += part['objs']
                    break
        scene_tmp = trimesh.Scene()
        for obj in os.listdir('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/textured_objs'):
            if obj.find('.obj') == -1: continue
            if obj[:-4] not in mobility_meshs: continue
            scene_tmp.add_geometry(trimesh.load('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' + model_id + '/textured_objs/' + obj,process=False))
        
        if mobility_part['joint'] == 'slider':
            #print(mobility_meshs)
            # for a slider joint, we simply move it to the max along the axis, and calculate its bbox.
            direction = np.array(mobility_part['jointData']['axis']['direction'])
            limit_1 = mobility_part['jointData']['limit']['a']
            limit_2 = mobility_part['jointData']['limit']['b']
            #print(limit_2)
            translation_1 = direction * limit_1
            translation_2 = direction * limit_2 - translation_1

            #scene_tmp.apply_transform(trimesh.transformations.translation_matrix(translation_1))
            #bbox1 = scene_tmp.bounding_box

            # scene_tmp.apply_transform(trimesh.transformations.translation_matrix(translation_2))
            # bbox2 = scene_tmp.bounding_box

            # scene.add_geometry(bbox1)
            # scene.add_geometry(bbox2)
            extents = scene_tmp.bounding_box_oriented.extents
            center = (scene_tmp.bounds[1] + scene_tmp.bounds[0]) / 2
            box_tmp = np.concatenate([orig_scene_center,orig_scene_extents,center,extents,np.array([0]),direction,np.array([limit_1,limit_2])])
            boxes_ret.append(box_tmp)

        if mobility_part['joint'] == 'hinge':
            # for a hinge joint, we rotate it to the max and calculate the bbox.
           # print(mobility_meshs)
            orig = np.array(mobility_part['jointData']['axis']['origin'])
            direction = np.array(mobility_part['jointData']['axis']['direction'])
            limit_1 = mobility_part['jointData']['limit']['a']
            limit_2 = mobility_part['jointData']['limit']['b']
            #print(limit_2)
            #if limit_2 > 90: limit_2 = 90
            #if limit_2 < -90: limit_2 = -90
            
            #rotation_a = trimesh.transformations.rotation_matrix(limit_1 / 180 * math.pi, direction,orig)
            #scene_tmp.apply_transform(rotation_a)

            # bbox1 = scene_tmp.bounding_box

            # rotation_b = trimesh.transformations.rotation_matrix((limit_2 - limit_1 )/ 180 * math.pi, direction,orig)
            # scene_tmp.apply_transform(rotation_b)
            # bbox2 = scene_tmp.bounding_box

            # scene.add_geometry(bbox1)
            # scene.add_geometry(bbox2)
            extents = scene_tmp.bounding_box_oriented.extents
            center = (scene_tmp.bounds[1] + scene_tmp.bounds[0]) / 2
            box_tmp = np.concatenate([orig_scene_center,orig_scene_extents,center,extents,np.array([1]),orig,direction,np.array([limit_1,limit_2])])
            #print(box_tmp)
            boxes_ret.append(box_tmp)
    return boxes_ret
    