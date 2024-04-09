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
import json
from common.structural_triangulation import *
#import robust_loss_pytorch.general
import logging
import sys
import optuna
from multiprocessing import Process
set_seed()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.GPU)
args = parse_args()
# dataset_path = '../MHFormer/dataset/data_3d_h36m.npz'
#dataset_path = "./data/data_3d_h36m.npz"

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

my_data_pth = path_dataset+args.data_name+".npz" ####one example:t1_o1_ex7
my_data = np.load(my_data_pth, allow_pickle=True)
my_data = dict(my_data)
actions = []
N_frame_action_dict = {}

f = open('data/bones_length_hall6_2d_pose_structure.json', )
bones_prior_dict = json.load(f)



count_subj=0
gt_bones_lens={}
processed_lens = np.load("data/bones_length_hall6_triang_measure_16_cams_h36m_struct.npy")

for sub_id in range(processed_lens.shape[0]):
    gt_bones_lens[sub_id] = processed_lens[sub_id]
print('gt_bones_lens:', gt_bones_lens)
struct_triang_results = {}
keypoints_new = {}
for sub in operators:
    # if sub!=5:
    #     continue
    keypoints['S{}'.format(sub)] = my_data['S{}'.format(sub)].item()
    ############ coco to h36m ######################

    #remove [S4][task1_example6] and [S4][task3_example6] from keypoints_new dict
    # if sub == 4:
    #     keypoints_new['S{}'.format(sub)].pop('task1_example6')
    #     keypoints_new['S{}'.format(sub)].pop('task3_example6')
    print('sub:', sub)
    keypoints_new['S{}'.format(sub)]={}
    struct_triang_results['S{}'.format(sub)] = {}
    t_i=0

    for task in tasks:
        # if task != 1:
        #     continue
        if task==0:
            continue
        for example in examples:
            # if example!=2:
            #     continue
            print('task:', task, 'example:', example)
            """path_meta_data = path_meta_data + 'task'+str(task) + '/operator' + str(sub) + '/example' + str(example) + '/dataset/'
            path_meta_data = path_meta_data + 'task'+str(task) + '_' + str(sub) + '_example' + str(example)+'_metadata.npz'
            meta_data_load = np.load(path_meta_data, allow_pickle=True)
            meta_data_load = dict(meta_data_load)
            meta_data['S{}'.format(sub)] = meta_data_load['S{}'.format(sub)].item()"""
            # for i in range(len(cfg.HALL6_DATA.TRAIN_CAMERAS)): #4 views
            if sub==4 and example==6:
                if task==1 or task==3:
                    continue
            if 'task{}_example{}'.format(task, example) in keypoints['S{}'.format(sub)]:
                keypoints_new['S{}'.format(sub)]["task{}_example{}".format(task, example)] = []
                struct_triang_results['S{}'.format(sub)]["task{}_example{}".format(task, example)] = []
                curr_ex = []
                min_len=min([len(l) for l in keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)]])
                for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                    #print('Processing view {}......'.format(i))
                    #print(keypoints_new['S{}'.format(sub)].keys())
                    keypoints_new['S{}'.format(sub)]["task{}_example{}".format(task, example)].append(copy.deepcopy(keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len, :, :]))
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,0,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,11,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,12,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,8,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,5,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,6,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,7,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,0,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,8,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,10,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,3,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,4,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,9,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,8,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,10,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],:] = keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:min_len,[12, 14, 16, 11, 13, 15, 5, 7, 9, 6, 8, 10],:]
                    curr_ex.append(keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i])


                actions.append('task{}_example{}'.format(task, example))
                n_frame_current_ex = keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i].shape[0]
                N_frame_action_dict[n_frame_current_ex]='task{}_example{}'.format(task, example)
                N=len(cfg.HALL6_DATA.TRAIN_CAMERAS)
                curr_ex = np.array(curr_ex)
                for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                    struct_triang_results['S{}'.format(sub)]["task{}_example{}".format(task, example)].append([])
                for t in range(curr_ex.shape[1]):
                    frame = curr_ex[:,t,:]

                    inputs_2d_gt = frame[..., :2]
                    inputs_2d_pre = frame[..., 2:4]
                    cam_3d = frame[..., 4:7]
                    conf = np.squeeze(frame[..., 7:8])

                    lens=np.expand_dims(gt_bones_lens[sub-1], 1)
                    view_list = cfg.HALL6_DATA.TRAIN_CAMERAS
                    prj_mat=HumanCam.camera_set['S{}'.format(sub)]["T" + str(task)]["E" + str(example)]['prj_mat'][view_list].detach().cpu().numpy()

                    try:
                        pose_3D = Pose3D_inference(N, create_human_tree(data_type="human36m"), inputs_2d_pre , conf, lens, prj_mat, "ST", 9)
                        # repeat the 3D pose to all the views
                        pose_3D = np.repeat(np.expand_dims(pose_3D, 0), N, axis=0)
                        for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                            #print(inputs_2d_pre.shape, pose_3D.shape, conf.shape)
                            data_pt = np.concatenate([inputs_2d_gt, inputs_2d_pre, pose_3D, np.expand_dims(conf, -1)], axis=-1)
                            struct_triang_results['S{}'.format(sub)]["task{}_example{}".format(task, example)][i].append(data_pt[i, :, :])
                    except:
                        for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                            # add nan to the struct_triang_results
                            print('triangulation failed')
                            # create a nan array of shape (N, 17, 3)
                            a = np.empty((N,inputs_2d_pre.shape[1], 3))
                            a[:] = np.nan
                            data_pt = np.concatenate([inputs_2d_gt, inputs_2d_pre, a, np.expand_dims(conf, -1)], axis=-1)
                            struct_triang_results['S{}'.format(sub)]["task{}_example{}".format(task, example)][i].append(data_pt[i, :, :])
                for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                    struct_triang_results['S{}'.format(sub)]["task{}_example{}".format(task, example)][i] = np.array(struct_triang_results['S{}'.format(sub)]["task{}_example{}".format(task, example)][i])

        t_i += 1
# save the struct_triang_results
np.savez(path_dataset + '/'+args.triang_out_name+".npz", **struct_triang_results)
