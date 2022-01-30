"""
    This file contains all helper utility functions.
"""

import os
import sys
import math
import glob
import importlib
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import trimesh, configparser
from pyquaternion import Quaternion
import h5py
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import kornia.geometry as tgm
# from lapsolver import solve_dense
# from natsort import natsorted, ns

# PI_ = torch.acos(torch.zeros(1)).item() * 2

# def worker_init_fn(worker_id):
#     """ The function is designed for pytorch multi-process dataloader.
#         Note that we use the pytorch random generator to generate a base_seed.
#         Please try to be consistent.
#         References:
#             https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
#     """
#     base_seed = torch.IntTensor(1).random_().item()
#     #print(worker_id, base_seed)
#     np.random.seed(base_seed + worker_id)

# def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None):
#     if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
#         raise ValueError('Number of models, model names, or optimizers does not match.')

#     for model, model_name in zip(models, model_names):
#         if model is not None:
#             filename = f'net_{model_name}.pth'
#             if prepend_epoch:
#                 filename = f'{epoch}_' + filename
#             torch.save(model.state_dict(), os.path.join(dirname, filename))

#     if optimizers is not None:
#         filename = 'checkpt.pth'
#         if prepend_epoch:
#             filename = f'{epoch}_' + filename
#         checkpt = {'epoch': epoch}
#         for opt, optimizer_name in zip(optimizers, optimizer_names):
#             if opt is not None:
#                 checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
#         torch.save(checkpt, os.path.join(dirname, filename))

# def load_checkpoint(models, model_names, dirname, epoch=None, device="cuda:0", optimizers=None, optimizer_names=None, strict=True):
#     if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
#         raise ValueError('Number of models, model names, or optimizers does not match.')

#     for model, model_name in zip(models, model_names):
#         filename = f'net_{model_name}.pth'
#         if epoch is not None:
#             filename = f'{epoch}_' + filename
#         state_dict = torch.load(os.path.join(dirname, filename), map_location=torch.device(device))
#         current_state_dict = model.state_dict()
#         firstkey = [k for k in current_state_dict.keys()]
#         if firstkey[0].find('module.')>=0:
#             from collections import OrderedDict
#             new_state_dict = OrderedDict()
#             # state_dict = torch.load(os.path.join(dirname, filename))
#             for k, v in state_dict.items():
#                 name = 'module.' + k # remove `module.`
#                 new_state_dict[name] = v
#             model.load_state_dict(new_state_dict, strict=strict)
#         else:
#             model.load_state_dict(state_dict, strict=strict)

#         if sys.version_info >= (3,0):
#             not_initialized = set()
#         else:
#             from sets import Set
#             not_initialized = Set()
#         for k,v in current_state_dict.items():
#             if k not in state_dict or v.size() != state_dict[k].size():
#                 not_initialized.add(k.split('.')[0])
#         print(sorted(not_initialized))

#     start_epoch = 0
#     if optimizers is not None:
#         filename = 'checkpt.pth'
#         if epoch is not None:
#             filename = f'{epoch}_' + filename
#         filename = os.path.join(dirname, filename)
#         if os.path.exists(filename):
#             checkpt = torch.load(filename)
#             start_epoch = checkpt['epoch']
#             for opt, optimizer_name in zip(optimizers, optimizer_names):
#                 if opt is not None:
#                     opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
#             print(f'resuming from checkpoint {filename}')
#         else:
#             response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
#             if response != 'y':
#                 sys.exit()

#     return start_epoch

# def optimizer_to_device(optimizer, device):
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if torch.is_tensor(v):
#                 state[k] = v.to(device)

# def optimizer_to_device1(optimizer, device):
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if torch.is_tensor(v):
#                 state[k] = v.cuda(device)

# def vrrotvec2mat(rotvector):
#     s = math.sin(rotvector[3])
#     c = math.cos(rotvector[3])
#     t = 1 - c
#     x = rotvector[0]
#     y = rotvector[1]
#     z = rotvector[2]
#     m = rotvector.new_tensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
#     return m

# def get_model_module(model_version):
#     importlib.invalidate_caches()
#     return importlib.import_module(model_version)

# # row_counts, col_counts: row and column counts of each distance matrix (assumed to be full if given)
# def linear_assignment(distance_mat, row_counts=None, col_counts=None):
#     batch_ind = []
#     row_ind = []
#     col_ind = []
#     for i in range(distance_mat.shape[0]):
#         # print(f'{i} / {distance_mat.shape[0]}')

#         dmat = distance_mat[i, :, :]
#         if row_counts is not None:
#             dmat = dmat[:row_counts[i], :]
#         if col_counts is not None:
#             dmat = dmat[:, :col_counts[i]]

#         # rind, cind = linear_sum_assignment(dmat.to('cpu').detach().numpy())
#         rind, cind = solve_dense(dmat.to('cpu').detach().numpy())
#         rind = list(rind)
#         cind = list(cind)
#         # print(dmat)
#         # print(rind)
#         # print(cind)

#         if len(rind) > 0:
#             rind, cind = zip(*sorted(zip(rind, cind)))
#             rind = list(rind)
#             cind = list(cind)

#         # complete the assignemnt for any remaining non-active elements (in case row_count or col_count was given),
#         # by assigning them randomly
#         #if len(rind) < distance_mat.shape[1]:
#         #    rind.extend(set(range(distance_mat.shape[1])).difference(rind))
#         #    cind.extend(set(range(distance_mat.shape[1])).difference(cind))

#         batch_ind += [i]*len(rind)
#         row_ind += rind
#         col_ind += cind

#     return batch_ind, row_ind, col_ind

# def object_batch_boxes(objects, max_box_num):
#     box_num = []
#     boxes = torch.zeros(len(objects), 12, max_box_num)
#     for oi, obj in enumerate(objects):
#         obj_boxes = obj.boxes()
#         box_num.append(len(obj_boxes))
#         if box_num[-1] > max_box_num:
#             print(f'WARNING: too many boxes in object, please use a dataset that does not have objects with too many boxes, clipping the object for now.')
#             box_num[-1] = max_box_num
#             obj_boxes = obj_boxes[:box_num[-1]]
#         obj_boxes = [o.view(-1, 1) for o in obj_boxes]
#         boxes[oi, :, :box_num[-1]] = torch.cat(obj_boxes, dim=1)

#     return boxes, box_num

# # out shape: (label_count, in shape)
# def one_hot(inp, label_count):
#     out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
#     out[inp.view(-1), torch.arange(out.shape[1])] = 1
#     out = out.view((label_count,) + inp.shape)
#     return out

# def collate_feats(b):
#     return list(zip(*b))

# def export_ply_with_label(out, v, l):
#     num_colors = len(colors)
#     with open(out, 'w') as fout:
#         fout.write('ply\n')
#         fout.write('format ascii 1.0\n')
#         fout.write('element vertex '+str(v.shape[0])+'\n')
#         fout.write('property float x\n')
#         fout.write('property float y\n')
#         fout.write('property float z\n')
#         fout.write('property uchar red\n')
#         fout.write('property uchar green\n')
#         fout.write('property uchar blue\n')
#         fout.write('end_header\n')

#         for i in range(v.shape[0]):
#             cur_color = colors[l[i]%num_colors]
#             fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], \
#                     int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)))

# def load_pts(fn):
#     with open(fn, 'r') as fin:
#         lines = [item.rstrip() for item in fin]
#         pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
#         return pts

# def export_pts(out, v):
#     with open(out, 'w') as fout:
#         for i in range(v.shape[0]):
#             fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

# def load_obj(fn):
#     fin = open(fn, 'r')
#     lines = [line.rstrip() for line in fin]
#     fin.close()

#     vertices = []; faces = [];
#     for line in lines:
#         if line.startswith('v '):
#             vertices.append(np.float32(line.split()[1:4]))
#         elif line.startswith('f '):
#             faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

#     f = np.vstack(faces)
#     v = np.vstack(vertices)
#     return v, f

# def export_obj(out, v, f):
#     with open(out, 'w') as fout:
#         for i in range(v.shape[0]):
#             fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
#         for i in range(f.shape[0]):
#             fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

# def color2mtl(colorfile):
#     from vis_utils import load_semantic_colors
#     from datav1 import Tree
#     filepath, fullflname = os.path.split(colorfile)
#     fname, ext = os.path.splitext(fullflname)
#     Tree.load_category_info(fname)

#     sem_colors = load_semantic_colors(filename=colorfile)
#     for sem in sem_colors:
#         sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)

#     mtl_fid = open(os.path.join(filepath, fname + '.mtl'), 'w')
#     for i in range(len(Tree.part_id2name)):
#         partname = Tree.part_id2name[i + 1]
#         color = sem_colors[partname]
#         mtl_fid.write('newmtl m_%s\nKd %f %f %f\nKa 0 0 0\n' % (partname.replace('/', '-'), color[0], color[1], color[2]))
#     mtl_fid.close()

# def qrot(q, v):
#     """
#     Rotate vector(s) v about the rotation described by quaternion(s) q.
#     Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
#     where * denotes any number of dimensions.
#     Returns a tensor of shape (*, 3).
#     """
#     assert q.shape[-1] == 4
#     assert v.shape[-1] == 3
#     assert q.shape[:-1] == v.shape[:-1]

#     original_shape = list(v.shape)
#     q = q.view(-1, 4)
#     v = v.view(-1, 3)

#     qvec = q[:, 1:]
#     uv = torch.cross(qvec, v, dim=1)
#     uuv = torch.cross(qvec, uv, dim=1)
#     return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

# # pc is N x 3, feat is 10-dim
# def transform_pc(pc, feat):
#     num_point = pc.size(0)
#     center = feat[:3]
#     shape = feat[3:6]
#     quat = feat[6:]
#     pc = pc * shape.repeat(num_point, 1)
#     pc = qrot(quat.repeat(num_point, 1), pc)
#     pc = pc + center.repeat(num_point, 1)
#     return pc

# # pc is N x 3, feat is B x 10-dim
# def transform_pc_batch(pc, feat, anchor=False, rot=True):
#     batch_size = feat.size(0)
#     num_point = pc.size(0)
#     pc = pc.repeat(batch_size, 1, 1)
#     center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
#     shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
#     quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
#     if not anchor:
#         pc = pc * shape
#     if rot:
#         pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
#     if not anchor:
#         pc = pc + center
#     return pc

# def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
#     r"""Convert an angle axis to a quaternion.
#     The quaternion vector has components in (x, y, z, w) format.
#     Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
#     Args:
#         angle_axis (torch.Tensor): tensor with angle axis.
#     Return:
#         torch.Tensor: tensor with quaternion.
#     Shape:
#         - Input: :math:`(*, 3)` where `*` means, any number of dimensions
#         - Output: :math:`(*, 4)`
#     Example:
#         >>> angle_axis = torch.rand(2, 3)  # Nx3
#         >>> quaternion = angle_axis_to_quaternion(angle_axis)  # Nx4
#     """
#     if not torch.is_tensor(angle_axis):
#         raise TypeError("Input type is not a torch.Tensor. Got {}".format(
#             type(angle_axis)))

#     if not angle_axis.shape[-1] == 3:
#         raise ValueError(
#             "Input must be a tensor of shape Nx3 or 3. Got {}".format(
#                 angle_axis.shape))
#     # unpack input and compute conversion
#     a0: torch.Tensor = angle_axis[..., 0:1]
#     a1: torch.Tensor = angle_axis[..., 1:2]
#     a2: torch.Tensor = angle_axis[..., 2:3]
#     theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

#     theta: torch.Tensor = torch.sqrt(theta_squared + 1e-12)
#     half_theta: torch.Tensor = theta * 0.5

#     mask: torch.Tensor = theta_squared > 0.0
#     ones: torch.Tensor = torch.ones_like(half_theta)

#     k_neg: torch.Tensor = 0.5 * ones
#     k_pos: torch.Tensor = torch.sin(half_theta) / (theta + 1e-12)
#     k: torch.Tensor = torch.where(mask, k_pos, k_neg)
#     w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

#     quaternion: torch.Tensor = torch.zeros_like(angle_axis)
#     quaternion[..., 0:1] += a0 * k
#     quaternion[..., 1:2] += a1 * k
#     quaternion[..., 2:3] += a2 * k
#     return torch.cat([w, quaternion], dim=-1)

# # pc is N x 3, feat is 10-dim
# def transform_pc_angle(pc, feat):
#     num_point = pc.size(0)
#     center = feat[:3]
#     shape = feat[3:6]
#     angle = feat[6:] * PI_

#     rotm = torch.eye(3)
#     rotm[0, 0] = torch.cos(angle.squeeze())
#     rotm[0, 2] = torch.sin(angle.squeeze())
#     rotm[2, 0] = -torch.sin(angle.squeeze())
#     rotm[2, 2] = torch.cos(angle.squeeze())

#     pc = pc * shape.repeat(num_point, 1)
#     pc = torch.mm(pc, rotm.transpose(0,1))
#     pc = pc + center.repeat(num_point, 1)
#     return pc

# def transform_pc_batch_angle(pc, feat, anchor=False):
#     batch_size = feat.size(0)
#     num_point = pc.size(0)
#     pc = pc.repeat(batch_size, 1, 1)
#     center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
#     shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
#     angle = feat[:, 6:] * PI_
#     rotm = torch.eye(3).unsqueeze(dim=0).repeat(batch_size, 1, 1).to(device = pc.device)
#     rotm[:, 0, 0] = torch.cos(angle.squeeze())
#     rotm[:, 0, 2] = torch.sin(angle.squeeze())
#     rotm[:, 2, 0] = -torch.sin(angle.squeeze())
#     rotm[:, 2, 2] = torch.cos(angle.squeeze())

#     # axis = torch.zeros_like(center)
#     # axis[:, :, 1] = 1.0
#     # axis_angle = axis * angle.unsqueeze(dim=1).repeat(1, num_point, 1)
#     # quat = angle_axis_to_quaternion(axis_angle.view(-1, 3))
#     if not anchor:
#         pc = pc * shape
#     pc = torch.bmm(pc, rotm.transpose(1,2))
#     # pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
#     if not anchor:
#         pc = pc + center
#     return pc

# def get_surface_reweighting(xyz, cube_num_point):
#     x = xyz[0]
#     y = xyz[1]
#     z = xyz[2]
#     # assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
#     np = cube_num_point // 6
#     out = torch.cat([(x*y).repeat(np*2), (y*z).repeat(np*2), (x*z).repeat(np*2)])
#     out = out / (out.sum() + 1e-12)
#     return out

# def get_surface_reweighting_batch(xyz, cube_num_point):
#     x = xyz[:, 0]
#     y = xyz[:, 1]
#     z = xyz[:, 2]
#     # assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
#     np = cube_num_point // 6
#     out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
#                      (y*z).unsqueeze(dim=1).repeat(1, np*2), \
#                      (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
#     out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
#     return out

# def gen_obb_mesh(obbs):
#     # load cube
#     cube_v, cube_f = load_obj('cube.obj')

#     all_v = []; all_f = []; vid = 0;
#     for pid in range(obbs.shape[0]):
#         p = obbs[pid, :]
#         center = p[0: 3]
#         lengths = p[3: 6]
#         dir_1 = p[6: 9]
#         dir_2 = p[9: ]

#         dir_1 = dir_1/np.linalg.norm(dir_1)
#         dir_2 = dir_2/np.linalg.norm(dir_2)
#         dir_3 = np.cross(dir_1, dir_2)
#         dir_3 = dir_3/np.linalg.norm(dir_3)

#         v = np.array(cube_v, dtype=np.float32)
#         f = np.array(cube_f, dtype=np.int32)
#         rot = np.vstack([dir_1, dir_2, dir_3])
#         v *= lengths
#         v = np.matmul(v, rot)
#         v += center

#         all_v.append(v)
#         all_f.append(f+vid)
#         vid += v.shape[0]

#     all_v = np.vstack(all_v)
#     all_f = np.vstack(all_f)
#     return all_v, all_f

# def sample_pc(v, f, n_points=2048):
#     mesh = trimesh.Trimesh(vertices=v, faces=f-1)
#     points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
#     return points

# def set_requires_grad(nets, requires_grad=False):
#     """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
#     Parameters:
#         nets (network list)   -- a list of networks
#         requires_grad (bool)  -- whether the networks require gradients or not
#     """
#     if not isinstance(nets, list):
#         nets = [nets]
#     for net in nets:
#         if net is not None:
#             for param in net.parameters():
#                 param.requires_grad = requires_grad

# def argpaser2file(args, name='example.ini'):
#     d = args.__dict__
#     cfpar = configparser.ConfigParser()
#     cfpar['default'] = {}
#     for key in sorted(d.keys()):
#         cfpar['default'][str(key)]=str(d[key])
#         print('%s = %s'%(key,d[key]))

#     with open(name, 'w') as configfile:
#         cfpar.write(configfile)

# def inifile2args(args, ininame='example.ini'):

#     config = configparser.ConfigParser()
#     config.read(ininame)
#     defaults = config['default']
#     result = dict(defaults)
#     # print(result)
#     # print('\n')
#     # print(args)
#     args1 = vars(args)
#     # print(args1)

#     args1.update({k: v for k, v in result.items() if v is not None})  # Update if v is not None

#     # print(args1)
#     args.__dict__.update(args1)

#     # print(args)

#     return args

# def neighbour2vdiff(neighbour, ref_V):
#     neighbour = neighbour - 1
#     pointnum = neighbour.shape[0]
#     maxdim = neighbour.shape[1]
#     vdiff = np.zeros((pointnum, maxdim, 3), dtype=np.float32)
#     for point_i in range(pointnum):
#         for j in range(maxdim):
#             curneighbour = neighbour[point_i][j]
#             if curneighbour == -1:
#                 break

#             vdiff[point_i][j] = ref_V[point_i] - ref_V[curneighbour]

#     return vdiff

# # ----------------------------------------------------------- ICP -------------------------------------------------------------

# def best_fit_transform(A, B):
#     '''
#     Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
#     Input:
#       A: Nxm numpy array of corresponding points
#       B: Nxm numpy array of corresponding points
#     Returns:
#       T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
#       R: mxm rotation matrix
#       t: mx1 translation vector
#     '''
#     #print(A.shape, B.shape)
#     assert A.shape == B.shape

#     # get number of dimensions
#     m = A.shape[1]

#     # translate points to their centroids
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     AA = A - centroid_A
#     BB = B - centroid_B

#     # rotation matrix
#     H = np.dot(AA.T, BB)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(Vt.T, U.T)

#     # special reflection case
#     if np.linalg.det(R) < 0:
#         Vt[m-1,:] *= -1
#         R = np.dot(Vt.T, U.T)

#     # translation
#     t = centroid_B.T - np.dot(R,centroid_A.T)

#     # homogeneous transformation
#     T = np.identity(m+1)
#     T[:m, :m] = R
#     T[:m, m] = t

#     return T, R, t


# def nearest_neighbor(src, dst):
#     '''
#     Find the nearest (Euclidean) neighbor in dst for each point in src
#     Input:
#         src: Nxm array of points
#         dst: Nxm array of points
#     Output:
#         distances: Euclidean distances of the nearest neighbor
#         indices: dst indices of the nearest neighbor
#     '''

#     assert src.shape == dst.shape

#     neigh = NearestNeighbors(n_neighbors=1)
#     neigh.fit(dst)
#     distances, indices = neigh.kneighbors(src, return_distance=True)
#     return distances.ravel(), indices.ravel()


# def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
#     '''
#     The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
#     Input:
#         A: Nxm numpy array of source mD points
#         B: Nxm numpy array of destination mD point
#         init_pose: (m+1)x(m+1) homogeneous transformation
#         max_iterations: exit algorithm after max_iterations
#         tolerance: convergence criteria
#     Output:
#         T: final homogeneous transformation that maps A on to B
#         distances: Euclidean distances (errors) of the nearest neighbor
#         i: number of iterations to converge
#     '''

#     assert A.shape == B.shape

#     # get number of dimensions
#     m = A.shape[1]

#     # make points homogeneous, copy them to maintain the originals
#     src = np.ones((m+1,A.shape[0]))
#     dst = np.ones((m+1,B.shape[0]))
#     src[:m,:] = np.copy(A.T)
#     dst[:m,:] = np.copy(B.T)

#     # apply the initial pose estimation
#     if init_pose is not None:
#         src = np.dot(init_pose, src)

#     prev_error = 0

#     for i in range(max_iterations):
#         # find the nearest neighbors between the current source and destination points
#         distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

#         # compute the transformation between the current source and nearest destination points
#         T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

#         # update the current source
#         src = np.dot(T, src)

#         # check error
#         mean_error = np.mean(distances)
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error

#     # calculate final transformation
#     T,_,_ = best_fit_transform(A, src[:m,:].T)

#     return T, distances, i


# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return min(len(d) for d in self.datasets)

# class MeshDataLoader():
#     def __init__(self, data_path):

#         if os.path.exists(os.path.join(data_path, 'cube_meshinfo.mat')):
#             meshinfo = os.path.join(data_path, 'cube_meshinfo.mat')
#         else:
#             meshinfo = os.path.join(data_path, 'floor_meshinfo_z.mat')
#         print(meshinfo)
#         meshdata = h5py.File(meshinfo, mode = 'r')

#         self.num_point = meshdata['neighbour'].shape[0]
#         self.edge_index = np.array(meshdata['edge_index']).astype('int64')
#         self.reconmatrix = np.array(meshdata['recon']).astype('float32')
#         self.ref_V = np.array(meshdata['ref_V']).astype('float32')
#         self.ref_F = np.array(meshdata['ref_F']).astype('int64')
#         self.vdiff = np.array(meshdata['vdiff']).astype('float32')
#         self.nb = np.array(meshdata['neighbour']).astype('int64')
#         self.part_num = 0
#         self.avg_feat = 0
#         print(len(self.ref_V))

# def add_meshinfo2conf(conf):

#     if 'data_path_a' in conf.__dict__.keys():
#         conf.meshinfo_a = MeshDataLoader(conf.data_path_a)

#         conf.meshinfo_a.edge_index = torch.tensor(conf.meshinfo_a.edge_index)#.to(conf.device)
#         conf.meshinfo_a.reconmatrix = torch.tensor(conf.meshinfo_a.reconmatrix)#.to(conf.device)
#         conf.meshinfo_a.ref_V = torch.tensor(conf.meshinfo_a.ref_V)#.to(conf.device)
#         conf.meshinfo_a.ref_F = conf.meshinfo_a.ref_F
#         conf.meshinfo_a.vdiff = torch.tensor(conf.meshinfo_a.vdiff)#.to(conf.device)
#         conf.meshinfo_a.nb = torch.tensor(conf.meshinfo_a.nb)#.to(conf.device)
#         conf.meshinfo_a.gpu = conf.gpu
#         # conf.meshinfo_a.device = conf.device
#         conf.meshinfo_a.point_num = conf.meshinfo_a.ref_V.shape[0]
#         conf.meshinfo_a.avg_feat = torch.tensor(conf.meshinfo_a.avg_feat)

#     if 'data_path_b' in conf.__dict__.keys():
#         conf.meshinfo_b = MeshDataLoader(conf.data_path_b)

#         conf.meshinfo_b.edge_index = torch.tensor(conf.meshinfo_b.edge_index)#.to(conf.device)
#         conf.meshinfo_b.reconmatrix = torch.tensor(conf.meshinfo_b.reconmatrix)#.to(conf.device)
#         conf.meshinfo_b.ref_V = torch.tensor(conf.meshinfo_b.ref_V)#.to(conf.device)
#         conf.meshinfo_b.ref_F = conf.meshinfo_b.ref_F
#         conf.meshinfo_b.vdiff = torch.tensor(conf.meshinfo_b.vdiff)#.to(conf.device)
#         conf.meshinfo_b.nb = torch.tensor(conf.meshinfo_b.nb)#.to(conf.device)
#         conf.meshinfo_b.gpu = conf.gpu
#         # conf.meshinfo_b.device = conf.device
#         conf.meshinfo_b.point_num = conf.meshinfo_b.ref_V.shape[0]
#         conf.meshinfo_b.avg_feat = torch.tensor(conf.meshinfo_b.avg_feat)

#     if 'data_path_a' not in conf.__dict__.keys() and 'data_path_b' not in conf.__dict__.keys():
#         conf.meshinfo = MeshDataLoader(conf.data_path)

#         conf.meshinfo.edge_index = torch.tensor(conf.meshinfo.edge_index).to(conf.device)
#         conf.meshinfo.reconmatrix = torch.tensor(conf.meshinfo.reconmatrix).to(conf.device)
#         conf.meshinfo.ref_V = torch.tensor(conf.meshinfo.ref_V).to(conf.device)
#         conf.meshinfo.ref_F = conf.meshinfo.ref_F
#         conf.meshinfo.vdiff = torch.tensor(conf.meshinfo.vdiff).to(conf.device)
#         conf.meshinfo.nb = torch.tensor(conf.meshinfo.nb).to(conf.device)
#         conf.meshinfo.gpu = conf.gpu
#         # conf.meshinfo.device = conf.device
#         conf.meshinfo.point_num = conf.meshinfo.ref_V.shape[0]

#     return conf


# class MeshDataLoaderv2():
#     def __init__(self, data_path):

#         print(data_path)
#         meshdata = h5py.File(data_path, mode = 'r')

#         self.num_point = meshdata['neighbour'].shape[0]
#         self.edge_index = np.array(meshdata['edge_index']).astype('int64')
#         self.reconmatrix = np.array(meshdata['recon']).astype('float32')
#         self.ref_V = np.array(meshdata['ref_V']).astype('float32')
#         self.ref_F = np.array(meshdata['ref_F']).astype('int64')
#         self.vdiff = np.array(meshdata['vdiff']).astype('float32')
#         self.nb = np.array(meshdata['neighbour']).astype('int64')
#         self.part_num = 0
#         self.avg_feat = 0
#         print(len(self.ref_V))


# def add_floormeshinfo2conf(conf):
#     if conf.data_path.endswith('.mat'):
#         matfile_name = conf.data_path
#     else:
#         matfile_name = os.path.join(conf.data_path, 'floor_meshinfo_z.mat')
#     conf.floor_meshinfo = MeshDataLoaderv2(matfile_name)

#     conf.floor_meshinfo.edge_index = torch.tensor(conf.floor_meshinfo.edge_index).cuda(conf.gpu)
#     conf.floor_meshinfo.reconmatrix = torch.tensor(conf.floor_meshinfo.reconmatrix).cuda(conf.gpu)
#     conf.floor_meshinfo.ref_V = torch.tensor(conf.floor_meshinfo.ref_V).cuda(conf.gpu)
#     conf.floor_meshinfo.ref_F = conf.floor_meshinfo.ref_F
#     conf.floor_meshinfo.vdiff = torch.tensor(conf.floor_meshinfo.vdiff).cuda(conf.gpu)
#     conf.floor_meshinfo.nb = torch.tensor(conf.floor_meshinfo.nb).cuda(conf.gpu)
#     conf.floor_meshinfo.gpu = conf.gpu
#     # conf.floor_meshinfo.device = conf.gpu
#     conf.floor_meshinfo.point_num = conf.floor_meshinfo.ref_V.shape[0]

#     return conf


# def add_objectmeshinfo2conf(conf):

#     conf.object_meshinfo = MeshDataLoaderv2(os.path.join(conf.data_path, 'object_meshinfo.mat'))

#     conf.object_meshinfo.edge_index = torch.tensor(conf.object_meshinfo.edge_index).cuda(conf.gpu)
#     conf.object_meshinfo.reconmatrix = torch.tensor(conf.object_meshinfo.reconmatrix).cuda(conf.gpu)
#     conf.object_meshinfo.ref_V = torch.tensor(conf.object_meshinfo.ref_V).cuda(conf.gpu)
#     conf.object_meshinfo.ref_F = conf.object_meshinfo.ref_F
#     conf.object_meshinfo.vdiff = torch.tensor(conf.object_meshinfo.vdiff).cuda(conf.gpu)
#     conf.object_meshinfo.nb = torch.tensor(conf.object_meshinfo.nb).cuda(conf.gpu)
#     conf.object_meshinfo.gpu = conf.gpu
#     # conf.object_meshinfo.device = conf.gpu
#     conf.object_meshinfo.point_num = conf.object_meshinfo.ref_V.shape[0]

#     return conf

# def weight_losses(losses, conf, sum = False):

#     if 'box' in losses.keys():
#         losses['box'] *= conf.loss_weight_box
#     if 'leaf' in losses.keys():
#         losses['leaf'] *= conf.loss_weight_leaf
#     if 'exists' in losses.keys():
#         losses['exists'] *= conf.loss_weight_exists
#     if 'semantic' in losses.keys():
#         losses['semantic'] *= conf.loss_weight_semantic
#     if 'anchor' in losses.keys():
#         losses['anchor'] *= conf.loss_weight_anchor
#     if 'feat' in losses.keys():
#         losses['feat'] *= conf.loss_weight_latent
#     if 'surf' in losses.keys():
#         losses['surf'] *= conf.loss_weight_geo
#     if 'center' in losses.keys():
#         losses['center'] *= conf.loss_weight_center
#     if 'acap' in losses.keys():
#         losses['acap'] *= conf.loss_weight_dggeo
#     if 'kldiv' in losses.keys():
#         losses['kldiv'] *= conf.loss_weight_kldiv
#     if 'node_nll' in losses.keys():
#         losses['node_nll'] *= conf.loss_weight_node_kldiv
#     if 'obj_tree' in losses.keys():
#         losses['obj_tree'] *= conf.loss_weight_obj_tree
#     if 'located' in losses.keys():
#         losses['located'] *= conf.loss_weight_obj_located
#     if 'edge_exists' in losses.keys():
#         losses['edge_exists'] *= conf.loss_weight_edge_exists
#     if 'sym' in losses.keys():
#         losses['sym'] *= conf.loss_weight_sym
#     if 'adj' in losses.keys():
#         losses['adj'] *= conf.loss_weight_adj
#     if 'aux' in losses.keys():
#         losses['aux'] *= conf.loss_weight_sym
#     if 'match' in losses.keys():
#         losses['match'] *= conf.loss_weight_leaf
#     if 'room_tree' in losses.keys():
#         losses['room_tree'] *= conf.loss_weight_obj_tree
#     if sum:
#         total_loss = 0
#         for loss in losses.values():
#             total_loss += loss
#         return total_loss
#     else:
#         return losses

# # def weights_init(m):
# #     classname = m.__class__.__name__
# #     if classname.find('Conv') != -1:
# #         m.weight.data.normal_(0.0, 1e-6)
# #         m.bias.data.fill_(0)
# #     elif classname.find('BatchNorm2d') != -1:
# #         m.weight.data.normal_(0.0, 0.001)
# #         m.bias.data.fill_(0)

# def tree_find(tree, value):
#     def tree_rec(tree, iseq):
#         if isinstance(tree, list):
#             for i, child in enumerate(tree):
#                 r = tree_rec(child, iseq + [i])
#                 if r is not None:
#                     return r
#         elif tree == value:
#             return iseq
#         else:
#             return None

#     return tree_rec(tree, [])

# def weights_init(m):
#     if hasattr(m, 'weight'):
#         m.weight.data.fill_(0)#normal_(0.0, 0.000001)
#     if hasattr(m, 'bias'):
#         m.bias.data.fill_(0)# + 1e-6

# def backup_code(path = '../data/codebak'):
#     import time, random
#     timecurrent = time.strftime('%m%d%H%M', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
#     print("code backuping... " + timecurrent, end=' ', flush=True)
#     os.makedirs(os.path.join(path, timecurrent))
#     os.system('cp -r ./* %s/\n' % os.path.join(path, timecurrent))
#     print("DONE!")

# def mesh_to_obb(v):
#     # mesh = trimesh.Trimesh(vertices=v, faces=f-1)
#     points = v
#     pca = PCA()
#     pca.fit(points)
#     pcomps = pca.components_

#     points_local = np.matmul(pcomps, points.transpose()).transpose()

#     all_max = points_local.max(axis=0)
#     all_min = points_local.min(axis=0)

#     center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
#     size = all_max - all_min

#     xdir = pcomps[0, :]
#     ydir = pcomps[1, :]

#     return np.hstack([center, size, xdir, ydir]).astype(np.float32)


# def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
#     """
#     Args:
#         sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
#         targets: true value, one-hot-like vector of size [N,C]
#         pos_weight: Weight for postive sample
#     """
#     if not (targets.size() == sigmoid_x.size()):
#         raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

#     loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

#     if weight is not None:
#         loss = loss * weight

#     if not reduce:
#         return loss
#     elif size_average:
#         return loss.mean()
#     else:
#         return loss.sum()

# class WeightedBCELoss(torch.nn.Module):
#     def __init__(self, pos_weight=1, weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True):
#         """
#         Args:
#             pos_weight = Weight for postive samples. Size [1,C]
#             weight = Weight for Each class. Size [1,C]
#             PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
#             WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
#         """
#         super().__init__()

#         self.register_buffer('weight', weight)
#         self.register_buffer('pos_weight', pos_weight)
#         self.size_average = size_average
#         self.reduce = reduce
#         self.PosWeightIsDynamic = PosWeightIsDynamic

#     def forward(self, input, target):
#         # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
#         if self.PosWeightIsDynamic:
#             positive_counts = target.sum(dim=0)
#             nBatch = len(target)
#             self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

#         if self.weight is not None:
#             # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
#             return weighted_binary_cross_entropy(input, target,
#                                                  self.pos_weight,
#                                                  weight=self.weight,
#                                                  size_average=self.size_average,
#                                                  reduce=self.reduce)
#         else:
#             return weighted_binary_cross_entropy(input, target,
#                                                  self.pos_weight,
#                                                  weight=None,
#                                                  size_average=self.size_average,
#                                                  reduce=self.reduce)


# class StableBCELoss(torch.nn.modules.Module):
#        def __init__(self):
#              super(StableBCELoss, self).__init__()
#        def forward(self, input, target):
#              neg_abs = - input.abs()
#              loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
#              return loss.mean()


# def outline_to_real(v, f, wall_height = 2.5, wall_thickness = 0.125):
#     '''
#         According to floor outline, generate the floor mesh and wall
#         by jiamu
#         input: (floor outline vert, floor ouline face)
#         returns: (floor vert, floor face, wall vert, wall face)
#     '''
#     t = trimesh.Trimesh(vertices=v,faces=f,process=False)


#     vertices2d = t.vertices.copy()
#     vertices2d = t.vertices[:,[0,2]]

#     vertices2d = np.vstack((vertices2d,vertices2d[0,:].reshape(1,2)))
#     v = np.zeros((vertices2d.shape[0],3))
#     v[:,[0,2]] = vertices2d

#     p = trimesh.load_path(vertices2d)
#     p = p.simplify()
#     (vertices2d, faces) = p.triangulate()
#     vertices = np.zeros((vertices2d.shape[0],3))
#     vertices[:,[0,2]] = vertices2d

#     faces[:,[0,1,2]] = faces[:,[0,2,1]]

#     floor=trimesh.Trimesh(vertices = vertices, faces=faces)
#     # floor.export('xxxfloor.obj')

#     num_edges = v.shape[0]


#     vertices = np.vstack((v,v[0,:].reshape(1,3)))
#     v2 = vertices.copy()
#     for i in range(num_edges):
#         e1 = vertices[i + 1,:]-vertices[i,:]
#         if i == num_edges - 1:
#             e2 = vertices[1,:]-vertices[0,:]
#         else:
#             e2 = vertices[i + 2,:]-vertices[i + 1,:]
#         temp = np.cross(np.array([0,1,0]),e1)
#         temp2 = np.cross(np.array([0,1,0]),e2)
#         if i == num_edges - 1:
#             v2[0,:] = v2[0,:]-np.sign(temp+temp2)*0.1
#             v2[num_edges,:] = v2[num_edges,:]-np.sign(temp+temp2)*0.1
#         else:
#             v2[i+1,:] = v2[i+1,:]-np.sign(temp+temp2)*wall_thickness

#     vertices = vertices[0:vertices.shape[0] - 1,:]
#     v2 = v2[0:v2.shape[0] - 1,:]

#     height = np.array([0,wall_height,0]).reshape(1,3)
#     height = np.repeat(height,vertices.shape[0],axis=0)
#     save_v = np.vstack((vertices,vertices + height,v2,v2+height))
#     save_f = np.zeros((num_edges * 8,3))

#     for i in range(num_edges):
#         if i == 0:
#             bef = num_edges - 1
#         else:
#             bef = i-1

#         save_f[i,:] =  [i,num_edges+i,bef]
#         save_f[i+num_edges,:] =  [bef,num_edges+i,num_edges+bef]
#         save_f[i+num_edges*2,:] = [num_edges*3+i,num_edges*2+i,num_edges*2+bef]
#         save_f[i+num_edges*3,:] = [num_edges*3+i,num_edges*2+bef,num_edges*3+bef]

#         save_f[i+num_edges*4,:] = [num_edges*2+i,i,bef]
#         save_f[i+num_edges*5,:] = [num_edges*2+i,bef,num_edges*2+bef]
#         save_f[i+num_edges*6,:] = [num_edges+i,num_edges*3+i,num_edges+bef]
#         save_f[i+num_edges*7,:] = [num_edges+bef,num_edges*3+i,num_edges*3+bef]

#     save_f[:, [0,1,2]] = save_f[:, [0,2,1]]
#     return (floor.vertices.copy(), floor.faces.copy()+1, save_v, save_f+1)

# def export_wallfromcorner(vertex, edge, wall_height=2.5, wall_thickness=0.1):
#     neighbor = []
#     for i in range(vertex.shape[0]):
#         neighbor.append([])

#     for i in range(edge.shape[0]):
#         neighbor[edge[i, 0]].append(edge[i, 1])
#         neighbor[edge[i, 1]].append(edge[i, 0])
#     neighbor = np.array(neighbor)

#     v = [vertex[0,:]]
#     start_point_id = 0
#     idx = 0
#     flags = np.zeros(vertex.shape[0])
#     flags[idx] = 1
#     cout = 0
#     next_point_id = None
#     while True:
#         cout+=1
#         if cout > 100:
#             return None, None, None, None
#         # print(idx)
#         for i in neighbor[idx]:
#             if not flags[i]:
#                 next_point_id = i
#                 break
#         if next_point_id is None:
#             return None, None, None, None
#         # print(next_point_id)
#         # input()
#         v.append(vertex[next_point_id,:])
#         flags[next_point_id] = 1
#         idx = next_point_id
#         if idx == start_point_id:
#             break
#         if np.sum(flags) == vertex.shape[0]:
#             break
#         # for i in neighbor[next_point_id]:
#         #     if not flags[neighbor[next_point_id,i]]:
#         #         next_point_id = neighbor[next_point_id,i]
#         #         break

#     v = np.array(v)
#     # print(v)
#     # cc()
#     vertices2d = v[:,[0,2]]
#     vertices2d = np.vstack((vertices2d,vertices2d[0,:].reshape(1,2)))
#     temp = np.zeros((vertices2d.shape[0],3))
#     temp[:,[0,2]] = vertices2d

#     p = trimesh.load_path(vertices2d)
#     p = p.simplify()
#     (vertices2d, faces) = p.triangulate()
#     if len(vertices2d) < 1:
#         return None, None, None, None

#     vertices = np.zeros((vertices2d.shape[0],3))
#     vertices[:,[0,2]] = vertices2d
#     faces[:,[0,1,2]] = faces[:,[0,2,1]]
#     floor_mesh=trimesh.Trimesh(vertices = vertices, faces=faces)

#     vertices = floor_mesh.vertices

#     outer_edge = []
#     for face in floor_mesh.faces:
#         for i, ei in enumerate(face):
#             edge = [face[i-1], face[i]]
#             if edge in outer_edge:
#                 outer_edge.pop(outer_edge.index(edge))
#             elif [edge[1], edge[0]] in outer_edge:
#                 outer_edge.pop(outer_edge.index([edge[1], edge[0]]))
#             else:
#                 outer_edge.append(edge)
#     outer_edge_ori = np.array(outer_edge)

#     for f in floor_mesh.faces:
#         if np.setdiff1d(f,outer_edge_ori[0]).shape[0] == 1:
#             third_point = np.setdiff1d(f,outer_edge_ori[0])
#             break
#     # print(outer_edge_ori)
#     # cc()
#     idx1 = np.argmin(np.sum(np.power(v - vertices[outer_edge_ori[0,0],:],2),1))
#     idx2 = np.argmin(np.sum(np.power(v - vertices[outer_edge_ori[0,1],:],2),1))

#     # print(idx1,idx2)
#     # cc()
#     # print(vertices[outer_edge_ori[0,0],:],vertices[outer_edge_ori[0,1],:],v[np.max([idx1,idx2]),:], v[np.min([idx1,idx2]),:])
#     if np.max([idx1,idx2]) == v.shape[0] -1 and np.min([idx1,idx2]) == 0:
#         edge_vector = v[0,:] - v[-1,:]
#         # print(edge_vector)
#     else:
#         edge_vector = v[np.max([idx1,idx2]),:] - v[np.min([idx1,idx2]),:]
#     # print(vertices[third_point,:])
#     top_vector = np.cross(edge_vector, vertices[third_point,:] - v[np.min([idx1,idx2]),:])[0]
#     # print(top_vector)
#     # ccc()
#     top_vector = top_vector / np.linalg.norm(top_vector)
#     # print(idx1,idx2,top_vector)
#     # print(v)

#     num_edges = v.shape[0]

#     vertices = np.vstack((v,v[0,:].reshape(1,3)))
#     v2 = vertices.copy()
#     for i in range(num_edges):
#         e1 = vertices[i + 1,:]-vertices[i,:]
#         if i == num_edges - 1:
#             e2 = vertices[1,:]-vertices[0,:]
#         else:
#             e2 = vertices[i + 2,:]-vertices[i + 1,:]
#         temp = np.cross(top_vector,e1)
#         temp2 = np.cross(top_vector,e2)
#         # print(temp,temp2)
#         if i == num_edges - 1:
#             v2[0,:] = v2[0,:]-np.sign(temp+temp2)*wall_thickness
#             v2[num_edges,:] = v2[num_edges,:]-np.sign(temp+temp2)*wall_thickness
#         else:
#             # print(v2[i+1,:] - np.sign(temp+temp2)*wall_thickness)
#             v2[i+1,:] = v2[i+1,:]-np.sign(temp+temp2)*wall_thickness
#             # print(v2[i+1,:])

#     # print(v2)
#     vertices = vertices[0:vertices.shape[0] - 1,:]
#     v2 = v2[0:v2.shape[0] - 1,:]

#     height = np.array([0,wall_height,0]).reshape(1,3)
#     height = np.repeat(height,vertices.shape[0],axis=0)
#     save_v = np.vstack((vertices,vertices + height,v2,v2+height))

#     save_f = np.zeros((num_edges * 8,3)).astype('int64')

#     # print(save_v)

#     for i in range(num_edges):
#         if i == 0:
#             bef = num_edges - 1
#         else:
#             bef = i-1

#         save_f[i,:] =  [i,num_edges+i,bef]
#         save_f[i+num_edges,:] =  [bef,num_edges+i,num_edges+bef]
#         save_f[i+num_edges*2,:] = [num_edges*3+i,num_edges*2+i,num_edges*2+bef]
#         save_f[i+num_edges*3,:] = [num_edges*3+i,num_edges*2+bef,num_edges*3+bef]

#         save_f[i+num_edges*4,:] = [num_edges*2+i,i,bef]
#         save_f[i+num_edges*5,:] = [num_edges*2+i,bef,num_edges*2+bef]
#         save_f[i+num_edges*6,:] = [num_edges+i,num_edges*3+i,num_edges+bef]
#         save_f[i+num_edges*7,:] = [num_edges+bef,num_edges*3+i,num_edges*3+bef]


#     temp1 = np.cross(save_v[save_f[0,1],:] - save_v[save_f[0,0],:], save_v[save_f[0,2],:] - save_v[save_f[0,0],:])
#     temp1 = temp1 / np.linalg.norm(temp1)


#     temp2 = np.cross(top_vector,save_v[save_f[0,0],:] - save_v[save_f[0,2],:])
#     temp2 = temp2 / np.linalg.norm(temp2)
#     # print(temp1, temp2)
#     if sum(temp1) != sum(temp2):
#         # print('x')
#         save_f[:, [0,1,2]] = save_f[:, [0,2,1]]

#     return (floor_mesh.vertices.copy(), floor_mesh.faces.copy()+1, save_v, save_f+1)

def export_wallfromcorner1(vertex, edge, wall_height=2.5, wall_thickness=0.5):
    neighbor = []
    vertex = vertex[:int(vertex.shape[0] / 2)]
    for i in range(vertex.shape[0]):
        neighbor.append([])
    
    for i in range(edge.shape[0]):
        neighbor[edge[i, 0]].append(edge[i, 1])
        neighbor[edge[i, 1]].append(edge[i, 0])
    
    neighbor = np.array(neighbor)
    

    v = [vertex[0,:]]
    start_point_id = 0
    idx = 0
    flags = np.zeros(vertex.shape[0])
    flags[idx] = 1
    while True:
        # print(idx)
        for i in neighbor[idx]:
            if not flags[i]:
                next_point_id = i
                break
        
        # print(next_point_id)
        # input()
        v.append(vertex[next_point_id,:])
        flags[next_point_id] = 1
        idx = next_point_id
        if idx == start_point_id:
            break
        if np.sum(flags) == vertex.shape[0]:
            break
        # for i in neighbor[next_point_id]:
        #     if not flags[neighbor[next_point_id,i]]:
        #         next_point_id = neighbor[next_point_id,i]
        #         break

    v = np.array(v)
    # print(v)
    # cc()
    vertices2d = v[:,[0,2]]
    vertices2d = np.vstack((vertices2d,vertices2d[0,:].reshape(1,2)))
    temp = np.zeros((vertices2d.shape[0],3))
    temp[:,[0,2]] = vertices2d

    p = trimesh.load_path(vertices2d)
    p = p.simplify()
    (vertices2d, faces) = p.triangulate()
    vertices = np.zeros((vertices2d.shape[0],3))
    vertices[:,[0,2]] = vertices2d
    faces[:,[0,1,2]] = faces[:,[0,2,1]]
    floor_mesh=trimesh.Trimesh(vertices = vertices, faces=faces)
    
    vertices = floor_mesh.vertices
    
    outer_edge = []
    for face in floor_mesh.faces:
        for i, ei in enumerate(face):
            edge = [face[i-1], face[i]]
            if edge in outer_edge:
                outer_edge.pop(outer_edge.index(edge))
            elif [edge[1], edge[0]] in outer_edge:
                outer_edge.pop(outer_edge.index([edge[1], edge[0]]))
            else:
                outer_edge.append(edge)
    outer_edge_ori = np.array(outer_edge)

    for f in floor_mesh.faces:
        if np.setdiff1d(f,outer_edge_ori[0]).shape[0] == 1:
            third_point = np.setdiff1d(f,outer_edge_ori[0])
            break
    # print(outer_edge_ori)
    # cc()
    idx1 = np.argmin(np.sum(np.power(v - vertices[outer_edge_ori[0,0],:],2),1))
    idx2 = np.argmin(np.sum(np.power(v - vertices[outer_edge_ori[0,1],:],2),1))

    # print(idx1,idx2)
    # cc()
    # print(vertices[outer_edge_ori[0,0],:],vertices[outer_edge_ori[0,1],:],v[np.max([idx1,idx2]),:], v[np.min([idx1,idx2]),:])
    if np.max([idx1,idx2]) == v.shape[0] -1 and np.min([idx1,idx2]) == 0:
        edge_vector = v[0,:] - v[-1,:]
        # print(edge_vector)
    else:
        edge_vector = v[np.max([idx1,idx2]),:] - v[np.min([idx1,idx2]),:]
    # print(vertices[third_point,:])
    top_vector = np.cross(edge_vector, vertices[third_point,:] - v[np.min([idx1,idx2]),:])[0]
    # print(top_vector)
    # ccc()
    top_vector = top_vector / np.linalg.norm(top_vector)
    # print(idx1,idx2,top_vector)
    # print(v)
    
    num_edges = v.shape[0]
    
    
    vertices = np.vstack((v,v[0,:].reshape(1,3)))
    v2 = vertices.copy()
    for i in range(num_edges):
        e1 = vertices[i + 1,:]-vertices[i,:]
        if i == num_edges - 1:
            e2 = vertices[1,:]-vertices[0,:]
        else:
            e2 = vertices[i + 2,:]-vertices[i + 1,:]
        temp = np.cross(top_vector,e1)
        temp2 = np.cross(top_vector,e2)
        # print(temp,temp2)
        if i == num_edges - 1:
            v2[0,:] = v2[0,:]-np.sign(temp+temp2)*wall_thickness
            v2[num_edges,:] = v2[num_edges,:]-np.sign(temp+temp2)*wall_thickness
        else:
            # print(v2[i+1,:] - np.sign(temp+temp2)*wall_thickness)
            v2[i+1,:] = v2[i+1,:]-np.sign(temp+temp2)*wall_thickness
            # print(v2[i+1,:])

    # print(v2)
    vertices = vertices[0:vertices.shape[0] - 1,:]
    v2 = v2[0:v2.shape[0] - 1,:]

    height = np.array([0,wall_height,0]).reshape(1,3)
    height = np.repeat(height,vertices.shape[0],axis=0)
    save_v = np.vstack((vertices,vertices + height,v2,v2+height))
    
    save_f = np.zeros((num_edges * 8,3)).astype('int64')

    # print(save_v)

    for i in range(num_edges):
        if i == 0:
            bef = num_edges - 1
        else:
            bef = i-1

        save_f[i,:] =  [i,num_edges+i,bef]
        save_f[i+num_edges,:] =  [bef,num_edges+i,num_edges+bef]
        save_f[i+num_edges*2,:] = [num_edges*3+i,num_edges*2+i,num_edges*2+bef]
        save_f[i+num_edges*3,:] = [num_edges*3+i,num_edges*2+bef,num_edges*3+bef]

        save_f[i+num_edges*4,:] = [num_edges*2+i,i,bef]
        save_f[i+num_edges*5,:] = [num_edges*2+i,bef,num_edges*2+bef]
        save_f[i+num_edges*6,:] = [num_edges+i,num_edges*3+i,num_edges+bef]
        save_f[i+num_edges*7,:] = [num_edges+bef,num_edges*3+i,num_edges*3+bef]


    temp1 = np.cross(save_v[save_f[0,1],:] - save_v[save_f[0,0],:], save_v[save_f[0,2],:] - save_v[save_f[0,0],:])
    temp1 = temp1 / np.linalg.norm(temp1)


    temp2 = np.cross(top_vector,save_v[save_f[0,0],:] - save_v[save_f[0,2],:])
    temp2 = temp2 / np.linalg.norm(temp2)
    # print(temp1, temp2)
    if sum(temp1) != sum(temp2):
        # print('x')
        save_f[:, [0,1,2]] = save_f[:, [0,2,1]]

    return (floor_mesh.vertices.copy(), floor_mesh.faces.copy()+1, save_v, save_f+1)


def outline_to_real1(v,f,wall_height=2.5,wall_thickness=0.1):
    t = trimesh.Trimesh(vertices=v,faces=f,process=False)


    vertices2d = t.vertices.copy()
    vertices2d = t.vertices[:,[0,2]]

    vertices2d = np.vstack((vertices2d,vertices2d[0,:].reshape(1,2)))
    print(vertices2d)
    p = trimesh.load_path(vertices2d)
    
    p = p.simplify()
    
    (vertices2d,faces) = p.triangulate()
    
    vertices = np.zeros((vertices2d.shape[0],3))
    vertices[:,[0,2]] = vertices2d

    faces[:,[0,1,2]] = faces[:,[0,2,1]]

    floor=trimesh.Trimesh(vertices = vertices, faces=faces)

    num_edges = vertices.shape[0]

    vertices = np.vstack((vertices,vertices[0,:].reshape(1,3)))
    v2 = vertices.copy()
    for i in range(num_edges):
        e1 = vertices[i + 1,:]-vertices[i,:]
        if i == num_edges - 1:
            e2 = vertices[1,:]-vertices[0,:]
        else:
            e2 = vertices[i + 2,:]-vertices[i + 1,:]
        temp = -np.cross(np.array([0,1,0]),e1)
        temp2 = -np.cross(np.array([0,1,0]),e2)
        if i == num_edges - 1:
            v2[0,:] = v2[0,:]-np.sign(temp+temp2)*0.1
            v2[num_edges,:] = v2[num_edges,:]-np.sign(temp+temp2)*0.1
        else:
            v2[i+1,:] = v2[i+1,:]-np.sign(temp+temp2)*wall_thickness

    vertices = vertices[0:vertices.shape[0] - 1,:]
    v2 = v2[0:v2.shape[0] - 1,:]

    height = np.array([0,wall_height,0]).reshape(1,3)
    height = np.repeat(height,vertices.shape[0],axis=0)
    save_v = np.vstack((vertices,vertices + height,v2,v2+height))
    save_f = np.zeros((num_edges * 8,3))

    for i in range(num_edges):
        if i == 0:
            bef = num_edges - 1
        else:
            bef = i-1

        save_f[i,:] =  [i,num_edges+i,bef]
        save_f[i+num_edges,:] =  [bef,num_edges+i,num_edges+bef]
        save_f[i+num_edges*2,:] = [num_edges*2+i,num_edges*2+bef,num_edges*3+i]
        save_f[i+num_edges*3,:] = [num_edges*2+bef,num_edges*3+bef,num_edges*3+i]

        save_f[i+num_edges*4,:] = [i,bef,num_edges*2+i]
        save_f[i+num_edges*5,:] = [bef,num_edges*2+bef,num_edges*2+i]
        save_f[i+num_edges*6,:] = [num_edges+i,num_edges*3+i,num_edges+bef]
        save_f[i+num_edges*7,:] = [num_edges+bef,num_edges*3+i,num_edges*3+bef]

    return (floor.vertices.copy(),floor.faces.copy(),save_v,save_f)

def outline_to_real222(v,f,wall_height=2.5,wall_thickness=0.1,floor_has_thickness=True,floor_thickness=0.01):
    t = trimesh.Trimesh(vertices=v,faces=f,process=False)

    vertices2d = t.vertices.copy()
    vertices2d = t.vertices[:,[0,1]]

    vertices2d = np.vstack((vertices2d,vertices2d[0,:].reshape(1,2)))

    p = trimesh.load_path(vertices2d)
    p = p.simplify()
    (vertices2d,faces) = p.triangulate()
    vertices = np.zeros((vertices2d.shape[0],3))
    vertices[:,[0,2]] = vertices2d

    faces[:,[0,1,2]] = faces[:,[0,2,1]]

    floor=trimesh.Trimesh(vertices = vertices, faces=faces,process=False)

    num_edges = vertices.shape[0]

    vertices = np.vstack((vertices,vertices[0,:].reshape(1,3)))
    v2 = vertices.copy()
    for i in range(num_edges):
        e1 = vertices[i + 1,:]-vertices[i,:]
        if i == num_edges - 1:
            e2 = vertices[1,:]-vertices[0,:]
        else:
            e2 = vertices[i + 2,:]-vertices[i + 1,:]
        temp = -np.cross(np.array([0,1,0]),e1)
        temp2 = -np.cross(np.array([0,1,0]),e2)
        if i == num_edges - 1:
            v2[0,:] = v2[0,:]-np.sign(temp+temp2)*0.1
            v2[num_edges,:] = v2[num_edges,:]-np.sign(temp+temp2)*0.1
        else:
            v2[i+1,:] = v2[i+1,:]-np.sign(temp+temp2)*wall_thickness

    vertices = vertices[0:vertices.shape[0] - 1,:]
    v2 = v2[0:v2.shape[0] - 1,:]

    height = np.array([0,wall_height,0]).reshape(1,3)
    height = np.repeat(height,vertices.shape[0],axis=0)
    save_v = np.vstack((vertices,vertices + height,v2,v2+height))
    save_f = np.zeros((num_edges * 8,3))

    for i in range(num_edges):
        if i == 0:
            bef = num_edges - 1
        else:
            bef = i-1

        save_f[i,:] =  [i,num_edges+i,bef]
        save_f[i+num_edges,:] =  [bef,num_edges+i,num_edges+bef]
        save_f[i+num_edges*2,:] = [num_edges*2+i,num_edges*2+bef,num_edges*3+i]
        save_f[i+num_edges*3,:] = [num_edges*2+bef,num_edges*3+bef,num_edges*3+i]

        save_f[i+num_edges*4,:] = [i,bef,num_edges*2+i]
        save_f[i+num_edges*5,:] = [bef,num_edges*2+bef,num_edges*2+i]
        save_f[i+num_edges*6,:] = [num_edges+i,num_edges*3+i,num_edges+bef]
        save_f[i+num_edges*7,:] = [num_edges+bef,num_edges*3+i,num_edges*3+bef]


    if not floor_has_thickness:
        return (floor.vertices.copy(),floor.faces.copy(),save_v,save_f)
    else:

        floor_bottom=trimesh.Trimesh(vertices = vertices - np.array([0,floor_thickness,0]), faces=faces[:,[0,2,1]],process=False)

        vert_around = np.vstack((floor.vertices.copy() - np.array([0,floor_thickness,0]),floor.vertices.copy()))
        f_around = np.zeros_like(vert_around)
        num_v = floor.vertices.shape[0]
        for i in range(num_v):
            print(i)
            if i == 0:
                bef = num_v - 1
            else:
                bef = i-1
            f_around[i,:] =  [bef,i,num_v+i]
            f_around[i+num_v,:] = [num_v+bef,bef,num_v+i]
        mesh_around = trimesh.Trimesh(vertices=vert_around,faces=f_around,process=False)
        mesh_concat = trimesh.util.concatenate([floor, floor_bottom,mesh_around])
        return (mesh_concat.vertices.copy(),mesh_concat.faces.copy(),save_v,save_f)

# def get_vf_from_sceneroom(obj):
#     part_boxes, part_geos, edges, part_ids, part_sems = obj.graph4room()
#     v = []; f = []; c = []
#     # floor extraction
#     current_v_count = 0
#     for jj in range(len(part_boxes)):
#         # cur_corner = part_boxes[jj].numpy()
#         v.append(part_boxes[jj].view(1,-1))
#         c.append(torch.as_tensor([0, 255, 0], dtype = torch.int).view(1,-1))
#     # edge
#     for i in range(len(edges)):
#         f.append(torch.as_tensor([edges[i]['part_a'], edges[i]['part_b'], edges[i]['part_b']], dtype = torch.int).view(1,-1))
#     if len(v) > 0:
#         v = torch.cat(v, dim = 0)
#     if len(c) > 0:
#         c = torch.cat(c, dim = 0)
#     if len(f) > 0:
#         f = torch.cat(f, dim = 0)
#     return v, f, c


# import collections

# def dict_add(dict_1, dict_2):
#     for key1 in dict_1.keys():
#         dict_1[key1] += dict_2[key1]

#     return dict_1

# def find_lasted_ckpt(folder):
#     # print(folder)
#     ckptlist = natsorted(glob.glob(os.path.join(folder, '*_checkpt.pth')), alg=ns.IGNORECASE)
#     # print(ckptlist)
#     lasted_ckpt_epoch = int(ckptlist[-1].split('/')[-1].replace('_checkpt.pth', ''))
#     # print(lasted_chpt_epoch)
#     return lasted_ckpt_epoch

# def backup_code(path = '../data/codebak'):
#     import time, random
#     timecurrent = time.strftime('%m%d%H%M', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
#     print("code backuping... " + timecurrent, end=' ', flush=True)
#     os.makedirs(os.path.join(path, timecurrent))
#     os.system('cp -r ./*.py %s/\n' % os.path.join(path, timecurrent))
#     print("DONE!")


lst = os.listdir('H:\\partvae\\mesh')

i = 0
for fname in lst:
    print(fname)
    print(i)
    i += 1
    if fname.find('.bad') != -1 or os.path.exists('H:\\partvae\\mesh1\\' + fname+'.obj'): continue
    
    mesh = trimesh.load('H:\\partvae\\mesh\\' + fname,process=False)
   # mesh.show()
    v = mesh.vertices
    f = mesh.faces
    try:
        (_,_,v1,f1) = export_wallfromcorner1(v,f)
    except:
        continue
    m2 = trimesh.Trimesh(vertices = v1, faces = f1 - 1,process=False)
    with open('H:\\partvae\\mesh1\\' + fname+'.obj', 'w') as f:
        f.write(trimesh.exchange.obj.export_obj(m2, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8))