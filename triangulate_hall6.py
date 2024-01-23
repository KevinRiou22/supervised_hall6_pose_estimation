from gettext import translation
import numpy as np
import itertools
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import errno
import copy
import time
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from tensorboardX import SummaryWriter

from common.arguments import parse_args
from common.utils import deterministic_random, save_model, save_model_epoch
from common.camera import *
from common.multiview_model import get_models
from common.loss import *
from common.generators_triangulation import *
from common.data_augmentation_multi_view import *
from common.hall_6_dataset import Human36mCamera, Human36mDataset, Human36mCamDataset
from common.set_seed import *
from common.config import config as cfg
from common.config import reset_config, update_config
from common.vis import *
#import robust_loss_pytorch.general
import logging
import sys
import optuna
from multiprocessing import Process
set_seed()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.GPU)
args = parse_args()
# dataset_path = '../MHFormer/dataset/data_3d_h36m.npz'
dataset_path = "./data/data_3d_h36m.npz"

print('parsing args!')

print('parsed args!')
update_config(args.cfg)  ###config file->cfg
reset_config(cfg, args)  ###arg -> cfg
print(cfg)

print('p2d detector:{}'.format('gt_p2d' if cfg.DATA.USE_GT_2D else cfg.H36M_DATA.P2D_DETECTOR))
HumanCam = Human36mCamera(cfg)

"""keypoints = {}
for sub in [1, 5, 6, 7, 8, 9, 11]:
    if cfg.H36M_DATA.P2D_DETECTOR == 'cpn' or cfg.H36M_DATA.P2D_DETECTOR == 'gt':
        data_pth = 'data/h36m_sub{}.npz'.format(sub)
    elif cfg.H36M_DATA.P2D_DETECTOR == 'ada_fuse':
        data_pth = 'data/h36m_sub{}_ada_fuse.npz'.format(sub)

    keypoint = np.load(data_pth, allow_pickle=True)
    lst = keypoint.files
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]

kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = [kps_left, kps_right]"""
kps_left, kps_right = [1, 2, 3, 14, 15, 16], [4, 5, 6, 11, 12, 13]
joints_left, joints_right = [kps_left, kps_right]

my_l1_loss = nn.L1Loss()
keypoints = {}
keypoints_test = {}
my_keypoints = {}
keypoints_new = {}
meta_data = {}
#for sub in [1, 5, 6, 7, 8, 9, 11]:
path_dataset = "./data/"
path_meta_data = "./data/"
#operators = [1]

operators = [1, 2, 3, 4, 5]
tasks= [0, 1, 2, 3]
examples = list(range(1, 31))

my_data_pth = path_dataset+'hall6.npz' ####one example:t1_o1_ex7
my_data = np.load(my_data_pth, allow_pickle=True)
my_data = dict(my_data)
actions = []
N_frame_action_dict = {}
for sub in operators:
    keypoints['S{}'.format(sub)] = my_data['S1'].item()
    ############ coco to h36m ######################
    keypoints_new['S{}'.format(sub)] = copy.deepcopy(my_data['S1'].item())
    #remove [S4][task1_example6] and [S4][task3_example6] from keypoints_new dict
    if sub == 4:
        keypoints_new['S{}'.format(sub)].pop('task1_example6')
        keypoints_new['S{}'.format(sub)].pop('task3_example6')


    t_i=0
    for task in tasks:
        for example in examples:
            """path_meta_data = path_meta_data + 'task'+str(task) + '/operator' + str(sub) + '/example' + str(example) + '/dataset/'
            path_meta_data = path_meta_data + 'task'+str(task) + '_' + str(sub) + '_example' + str(example)+'_metadata.npz'
            meta_data_load = np.load(path_meta_data, allow_pickle=True)
            meta_data_load = dict(meta_data_load)
            meta_data['S{}'.format(sub)] = meta_data_load['S{}'.format(sub)].item()"""
            # for i in range(len(cfg.HALL6_DATA.TRAIN_CAMERAS)): #4 views
            if sub==4 and example==6:
                if task==1 or task==3:
                    continue
            if 'task{}_example{}'.format(task, example) in keypoints_new['S{}'.format(sub)]:
                for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                    print('Processing view {}......'.format(i))
                    print(keypoints_new['S{}'.format(sub)].keys())
                    #keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,0,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,11,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,12,:])/2
                    #keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,5,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,6,:])/2
                    #keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,7,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,0,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:])/2
                    #keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,10,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,3,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,4,:])/2
                    #keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,9,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,10,:])/2
                    #keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],:] = keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,[12, 14, 16, 11, 13, 15, 5, 7, 9, 6, 8, 10],:]
                actions.append('task{}_example{}'.format(task, example))
                n_frame_current_ex = keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i].shape[0]
                N_frame_action_dict[n_frame_current_ex]='task{}_example{}'.format(task, example)
        t_i += 1

# remove duplicate actions
actions = list(set(actions))

train_actions=actions
test_actions=actions
vis_actions=actions

action_frames = {}
for act in actions:
    action_frames[act] = 0
for k, v in N_frame_action_dict.items():
    action_frames[v] += k

def fetch(subjects, action_filter=None, parse_3d_poses=True, is_test=False, out_plus=False):  # for our own dataset
    out_poses_3d = []
    # out_poses_2d_view1 = []
    # out_poses_2d_view2 = []
    # out_poses_2d_view3 = []
    # out_poses_2d_view4 = []
    # num_views = len(cfg.HALL6_DATA.TRAIN_CAMERAS)
    num_views = 16  ## dataset contains 16 views in total
    out_poses_2d = {}

    for view_num in range(0, num_views):
        out_poses_2d[f'view{view_num}'] = []
    out_camera_params = []
    out_poses_3d = []
    out_subject_action = []
    used_cameras = cfg.HALL6_DATA.TEST_CAMERAS if is_test else cfg.HALL6_DATA.TRAIN_CAMERAS  ###0,1,2,3
    for subject in subjects:  ###['S1','S5','S6','S7','S8']

        for action in keypoints_new[subject].keys():  ###
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a) and len(action.split(a)[1]) < 3:
                        found = True
                        break
                if not found:
                    continue
            poses_2d = keypoints_new[subject][action]  # 4 list of (611, 17, 8)
            out_subject_action.append([subject, action])
            n_frames = poses_2d[0].shape[0]
            for view_num in range(0, num_views):
                out_poses_2d[f'view{view_num}'].append(poses_2d[view_num][:, :cfg.HALL6_DATA.NUM_JOINTS])
    final_pose = []
    for view_num in range(0, num_views):
        final_pose.append(out_poses_2d[f'view{view_num}'])
    if is_test is True:
        return final_pose
    else:
        return final_pose, out_subject_action

use_2d_gt = cfg.DATA.USE_GT_2D
receptive_field = cfg.NETWORK.TEMPORAL_LENGTH
pad = receptive_field // 2
causal_shift = 0
view_list = cfg.HALL6_DATA.TRAIN_CAMERAS

data_npy =  {}
data_npy_2d_from_3d = {}
data_npy_2d_gt = {}

if True:
    poses_train_2d, subject_action = fetch(cfg.HALL6_DATA.SUBJECTS_TRAIN, train_actions)
    train_generator = ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1, pad=pad, causal_shift=causal_shift,
                                       shuffle=False, augment=False, kps_left=None, kps_right=None,
                                       joints_left=None, joints_right=None, sub_act=subject_action,
                                       extra_poses_3d=None) if cfg.HALL6_DATA.PROJ_Frm_3DCAM == False else ChunkedGenerator(
        cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1, pad=pad, causal_shift=causal_shift, shuffle=False, augment=False,
        kps_left=None, kps_right=None, joints_left=None, joints_right=None)
    poses_train_2d, subject_action = fetch(cfg.HALL6_DATA.SUBJECTS_TRAIN, train_actions)

    data_aug = DataAug(cfg, add_view=cfg.TRAIN.NUM_AUGMENT_VIEWS)
    iters = 0
    msefun = torch.nn.L1Loss()
    num_train_views = len(cfg.HALL6_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
    last_loss = 0
    #########
    import json
    torch.autograd.set_detect_anomaly(True)
    id_eval = 0
    id_traj = 0
    batch_flip = [False for i in range(cfg.TRAIN.BATCH_SIZE)]
    for batch_2d, sub_action, batch_flip, current_id, frame_number in train_generator.next_epoch():
        inputs = torch.from_numpy(batch_2d.astype('float32'))
        inputs_2d_gt = inputs[..., :2, :]
        inputs_2d_pre = inputs[..., 2:4, :]
        #sub_action = [[valid_subject, act] for k in range(inputs.shape[0])]
        cam_3d = inputs[..., 4:7, :]
        vis = inputs[..., 7:8, :]
        inputs_3d_gt = cam_3d[:, pad:pad + 1]
        p3d_root = copy.deepcopy(inputs_3d_gt[:, :, :1])  # (B,T, 1, 3, N)
        inputs_3d_gt[:, :, 0] = 0
        inputs_3d_gt = inputs_3d_gt + p3d_root
        p3d_gt_abs = inputs_3d_gt + p3d_root
        if use_2d_gt:
            h36_inp = inputs_2d_gt
            vis = torch.ones(*vis.shape)
        else:
            h36_inp = inputs_2d_pre

        inp = h36_inp[..., view_list]  # B, T, V, C, N
        inp = torch.cat((inp, vis[..., view_list]), dim=-2)
        B = inp.shape[0]
        # if cfg.TEST.TRIANGULATE:
        #     trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t+1, :,:2, :], valid_subject, view_list)#B, T, J, 3, N)
        #     loss = 0
        #     for idx, view_idx in enumerate(view_list):
        #         loss_view_tmp = eval_metrc(cfg, trj_3d[..., idx], inputs_3d_gt[..., view_idx])
        #         loss += loss_view_tmp.item()
        #         action_mpjpe[act][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]
        #
        #     action_mpjpe[act][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
        #     continue
        prj_2dpre_to_3d, stats_sing_values, _ = HumanCam.p2d_cam3d_batch_with_root(h36_inp[:, pad:pad + 1, :, :, view_list], sub_action, view_list, debug=False, confidences=vis[:, pad:pad+1, :, :, view_list])
        prj_out_abs_to_2d = HumanCam.p3d_im2d_batch(prj_2dpre_to_3d, sub_action, view_list, with_distor=True, flip=batch_flip, gt_2d=inputs_2d_gt[:, pad:pad + 1, :, :].to(prj_2dpre_to_3d.device))

        # for id in current_id:
        #     if str(id[0]) not in data_npy_2d_gt.keys():
        #         data_npy_2d_gt[str(id[0])] = torch.zeros(
        #             (frame_number, 1, cfg.HALL6_DATA.NUM_JOINTS, 2, len(view_list)))
        #     else:
        #         pass
        # for i, id in enumerate(current_id):
        #     data_npy_2d_gt[str(id[0])][id[1], ...] = inp[:, pad:pad + 1, :, :2, :][i, ...].cpu()
        pose_2D_from3D_gt = prj_out_abs_to_2d.permute(0, 2, 3, 4, 1).contiguous()
        # print("frame number", frame_number)
        # print(current_id)
        # print("pose_2D_from3D_gt[i, ...].cpu() shape", pose_2D_from3D_gt[0, ...].cpu().shape)
        # print("inp[:, pad:pad + 1, :, :2, :][i, ...].cpu() shape", inp[:, pad:pad + 1, :, :2, :][0, ...].cpu().shape)
        # print("prj_2dpre_to_3d[i, ...].cpu() shape", prj_2dpre_to_3d[0, ...].cpu().shape)
        # print("inp[:, pad:pad + 1, :, -1, :][i, ...].cpu() shape", inp[:, pad:pad + 1, :, -1:, :][0, ...].cpu().shape)
        # print(inp[-1, pad:pad + 1, :, :2, :].cpu())
        for i, s_a in enumerate(sub_action):
            subject = s_a[0]
            action = s_a[1]
            if subject not in data_npy.keys():
                data_npy[subject] = {}
            if action not in data_npy[subject].keys():
                data_npy[subject][action] = []
                for v in range(len(view_list)):
                    data_npy[subject][action].append([])
            for v in view_list:
                # print(pose_2D_from3D_gt.shape)
                curr_data=torch.cat([pose_2D_from3D_gt[i, 0,:,:,v].cpu(), inp[:, :, :, :2, :][i, pad,:,:,v].cpu(), prj_2dpre_to_3d[i, 0,:,:,v].cpu(), inp[:, :, :, -1:, :][i, pad,:,:,v].cpu()], dim=-1)
                data_npy[subject][action][v].append(curr_data)

        # for i, id in enumerate(current_id):
        #     if frame_number > id[1]:
        #         data_npy[str(id[0])][id[1], ...] = torch.cat([pose_2D_from3D_gt[i, ...].cpu(), inp[:, pad:pad + 1, :, :2, :][i, ...].cpu(), prj_2dpre_to_3d[i, ...].cpu(), inp[:, pad:pad + 1, :, -1:, :][i, ...].cpu()], dim=-2)
        #
        # pose_2D_from3D_gt = prj_out_abs_to_2d.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, N, J. C)
        # for id in current_id:
        #     if str(id[0]) not in data_npy_2d_from_3d.keys():
        #         data_npy_2d_from_3d[str(id[0])] = torch.zeros( (frame_number, 1, cfg.HALL6_DATA.NUM_JOINTS, 2, len(view_list)))
        #     else:
        #         pass
        # for i, id in enumerate(current_id):
        #     data_npy_2d_from_3d[str(id[0])][id[1], ...] = pose_2D_from3D_gt[i, ...].cpu()
    print(view_list)
    for subject in data_npy.keys():
        for action in data_npy[subject].keys():
            for v in view_list:
                data_npy[subject][action][v] = torch.stack(data_npy[subject][action][v], dim=0).numpy()
    np.savez(path_dataset + '/triangulated_3D_with_distor_2D_structure.npz', **data_npy)
    # np.save(path_dataset + '/2d_from_triangulated_3D.npy', data_npy_2d_from_3d)
    # np.save(path_dataset + '/2d_pred.npy', data_npy_2d_gt)



