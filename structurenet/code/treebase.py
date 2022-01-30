import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from data import SceneTree

class OriTree(type):
    def __init__(cls, name, bases, attrs):
        cls.part_non_leaf_sem_names = []
        cls.part_name2id = dict()
        cls.part_id2name = dict()
        cls.part_name2cids = dict()
        cls.part_num_sem = None
        cls.root_sem = None
        cls.leaf_geos_box = None
        cls.leaf_geos_dg = None
        cls.leaf_geos_pts = None
        cls.cate_id = None

class Chair(SceneTree, metaclass=OriTree):
    pass
class Table(SceneTree, metaclass=OriTree):
    pass
class Storage_Furniture(SceneTree, metaclass=OriTree):
    pass
class Lamp(SceneTree, metaclass=OriTree):
    pass
class Bed(SceneTree, metaclass=OriTree):
    pass

