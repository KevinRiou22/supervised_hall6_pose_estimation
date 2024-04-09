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
from common.generators_kevin import *
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
def objective(trial):
    num_hyperparam_epochs = 5
    if cfg.TRAIN.USE_STD_LOSS:
        std_bone_len_weight = trial.suggest_float("std_bone_len_weight", 0, 1)
    if cfg.TRAIN.USE_BONE_DIR_VECT:
        direction_2d_bone_weight = trial.suggest_float("direction_2d_bone_weight", 0, 1)
    if cfg.TRAIN.USE_LOSS_BONES_2D:
        len_2d_bone_weight = trial.suggest_float("len_2d_bone_weight", 0, 1)
    if cfg.TRAIN.USE_2D_LOSS:
        joints_2d_weight = trial.suggest_float("joints_2d_weight", 0, 1)
    if cfg.TRAIN.USE_SYM_LOSS:
        symmetry_weight = trial.suggest_float("symmetry_weight", 0, 1)
    set_coord_sys_in_cam = False  # trial.suggest_categorical("set_coord_sys_in_cam", [True, False])


    trial_id = trial.number
    set_seed()
    # dataset_path = '../MHFormer/dataset/data_3d_h36m.npz'
    #dataset_path = "./data/data_3d_h36m.npz"
    print('parsing args!')

    print('parsed args!')
    update_config(args.cfg)  ###config file->cfg
    reset_config(cfg, args, trial_id=trial_id)  ###arg -> cfg
    print(cfg)
    if cfg.TRAIN.REGULARIZE_PARAMS:
        reg_params_weight = trial.suggest_float("reg_params_weight", 0, 1)
    if cfg.TRAIN.USE_BONES_PRIOR:
        human_prior_weight = trial.suggest_float("human_prior_weight", 0, 1)


    cfg.TRAIN.SET_COORD_SYS_IN_CAM = set_coord_sys_in_cam

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

    my_data_pth = path_dataset+"triangulated_3D_with_distor.npz"#'hall6.npz' ####one example:t1_o1_ex7
    my_data = np.load(my_data_pth, allow_pickle=True)
    my_data = dict(my_data)
    actions = []
    N_frame_action_dict = {}
    for sub in operators:
        keypoints['S{}'.format(sub)] = my_data['S{}'.format(sub)].item()
        print("subject {} is processing".format(sub))
        ############ coco to h36m ######################
        keypoints_new['S{}'.format(sub)] = copy.deepcopy(my_data['S{}'.format(sub)].item())
        #remove [S4][task1_example6] and [S4][task3_example6] from keypoints_new dict





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
                        """keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,0,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,11,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,12,:])/2
                        keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,5,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,6,:])/2
                        keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,7,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,0,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:])/2
                        keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,10,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,3,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,4,:])/2
                        keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,9,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,10,:])/2
                        keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],:] = keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,[12, 14, 16, 11, 13, 15, 5, 7, 9, 6, 8, 10],:]"""
                    actions.append('task{}_example{}'.format(task, example))
                    n_frame_current_ex = keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i].shape[0]
                    N_frame_action_dict[n_frame_current_ex]='task{}_example{}'.format(task, example)
            t_i += 1

    # remove duplicate actions
    actions = list(set(actions))

    train_actions=actions#[:int(0.8*len(actions))]
    test_actions=actions#[int(0.8*len(actions)):]
    vis_actions=actions#[int(0.8*len(actions)):]

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
    model, model_test = get_models(cfg)

    #####模型参数量、计算量(MACs)、inference time
    if cfg.VIS.VIS_COMPLEXITY:
        from thop import profile
        from thop import clever_format

        if args.eval:
            from ptflops import get_model_complexity_info
        #####模型参数量、计算量(MACs)
        receptive_field = 1
        model_test.eval()
        for i in range(1, 5):
            input = torch.randn(1, receptive_field, 17, 3, i)
            rotation = torch.randn(1, 3, 3, receptive_field, i, i)
            macs, params = profile(model_test, inputs=(input, rotation))
            macs, params = clever_format([macs, params], "%.3f")
            print('view: {} T: {} MACs:{} params:{}'.format(i, receptive_field, macs, params))
            if args.eval:
                flops, params = get_model_complexity_info(model_test, (receptive_field, 17, 3, i), as_strings=True,
                                                          print_per_layer_stat=False)
                print('Flops:{}, Params:{}'.format(flops, params))
        #####inference time
        infer_model = model_test.cuda()
        infer_model.eval()
        for receptive_field in [1, 27]:
            for i in range(1, 5):
                input = torch.randn(1, receptive_field, 17, 3, i).float().cuda()
                rotation = torch.randn(1, 3, 3, receptive_field, i, i).float().cuda()

                for k in range(100):
                    out = infer_model(input, rotation)

                N = 1000
                torch.cuda.synchronize()
                start_time = time.time()
                for n in range(N):
                    infer_model(input, rotation)
                torch.cuda.synchronize()
                end_time = time.time()
                print('n_frames:{} n_views: {}  time:{:.4f}'.format(receptive_field, i, (end_time - start_time) / N))
        exit()
    else:
        total_params = sum(p.numel() for p in model_test.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model_test.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    EVAL = args.eval
    ax_views = []

    if EVAL and cfg.VIS.VIS_3D:
        plt.ion()
        vis_tool = Vis(cfg, 2)

    def load_state(model_train, model_test):
        train_state = model_train.state_dict()
        test_state = model_test.state_dict()
        pretrained_dict = {k: v for k, v in train_state.items() if k in test_state}
        test_state.update(pretrained_dict)
        model_test.load_state_dict(test_state)

    if EVAL and not cfg.TEST.TRIANGULATE:
        chk_filename = cfg.TEST.CHECKPOINT
        print(chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print("('best_model' in checkpoint.keys()) : " + str(('best_model' in checkpoint.keys())))
        # input()
        model_checkpoint = checkpoint['model']  # if 'best_model' not in checkpoint.keys() else checkpoint['best_model']
        if model_checkpoint == None:
            model_checkpoint = checkpoint['model']
        train_checkpoint = model.state_dict()
        test_checkpoint = model_test.state_dict()
        for k, v in train_checkpoint.items():
            if k not in model_checkpoint.keys():
                continue
            checkpoint_v = model_checkpoint[k]
            if 'p_shrink.shrink' in k:
                if model_checkpoint[k].shape[0] == 32:
                    checkpoint_v = checkpoint_v[1::2]

            train_checkpoint[k] = checkpoint_v

        print('EVAL: This model was trained for {} epochs'.format(checkpoint['epoch']))
        model.load_state_dict(train_checkpoint)

    if True:
        if not cfg.DEBUG and (not args.eval or args.log):
            summary_writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        else:
            summary_writer = None

        poses_train_2d, subject_action = fetch(cfg.HALL6_DATA.SUBJECTS_TRAIN, train_actions)
        poses_test_2d, subject_test_action = fetch(cfg.HALL6_DATA.SUBJECTS_TRAIN, train_actions)
        #_, _, poses_3d_extra = fetch(cfg.HALL6_DATA.SUBJECTS_TRAIN, train_actions, out_plus=True)

        lr = cfg.TRAIN.LEARNING_RATE
        if cfg.TRAIN.RESUME:
            chk_filename = cfg.TRAIN.RESUME_CHECKPOINT
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            print('RESUME: This model was trained for {} epochs'.format(checkpoint['epoch']))
            model.load_state_dict(checkpoint['model'])
        if torch.cuda.is_available() and not cfg.TEST.TRIANGULATE:
            model = torch.nn.DataParallel(model).cuda()
            model_test = torch.nn.DataParallel(model_test).cuda()
        # adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device='cuda:0')
        # adaptive_bones = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device='cuda:0')
        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr,
                               amsgrad=True)  # + list(adaptive.parameters())+list(adaptive_bones.parameters())
        if cfg.TRAIN.RESUME:
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_result_epoch = checkpoint['best_epoch']
            best_state_dict = checkpoint['best_model']
            lr = checkpoint['lr']  # 0.0005#checkpoint['lr']
            best_result = 100
        else:
            epoch = 0
            best_result = 100
            best_state_dict = None
            best_result_epoch = 0

        lr_decay = cfg.TRAIN.LR_DECAY
        initial_momentum = 0.1
        final_momentum = 0.001
        train_generator = ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1, pad=pad, causal_shift=causal_shift,
                                           shuffle=True, augment=False, kps_left=kps_left, kps_right=kps_right,
                                           joints_left=joints_left, joints_right=joints_right, sub_act=subject_action,
                                           extra_poses_3d=None) if cfg.HALL6_DATA.PROJ_Frm_3DCAM == True else ChunkedGenerator(
            cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1, pad=pad, causal_shift=causal_shift, shuffle=True, augment=False,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
            sub_act=subject_action)

        test_generator = ChunkedGenerator(cfg.TEST.BATCH_SIZE, poses_test_2d, 1, pad=pad, causal_shift=causal_shift,
                                           shuffle=True, augment=False, kps_left=kps_left, kps_right=kps_right,
                                           joints_left=joints_left, joints_right=joints_right, sub_act=subject_test_action,
                                           extra_poses_3d=None) if cfg.HALL6_DATA.PROJ_Frm_3DCAM == True else ChunkedGenerator(
            cfg.TEST.BATCH_SIZE, poses_test_2d, 1, pad=pad, causal_shift=causal_shift, shuffle=True, augment=False,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
            sub_act=subject_test_action)

        print('** Starting.')

        data_aug = DataAug(cfg, add_view=cfg.TRAIN.NUM_AUGMENT_VIEWS)
        iters = 0
        msefun = torch.nn.L1Loss()
        num_train_views = len(cfg.HALL6_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        last_loss = 0
        #########
        import json

        print("using bones length priors!")
        # f = open('bone_priors_mean.json', )
        """f = open('bone_priors_mean.json', )
        gt_bones_lens = json.load(f)
    
        bones_means_ = []
        for i in gt_bones_lens.keys():
            bones_means_.append(gt_bones_lens[i])
        bones_means = torch.from_numpy(np.mean(np.array(bones_means_), axis=(0, 1))).cuda()
        print("bones_means : " + str(bones_means))
        bones = torch.from_numpy(np.array(cfg.HALL6_DATA.BONES)).cuda()
        print("bones : " + str(bones))"""


        print("using bones length priors!")
        #f = open('bone_priors_mean.json', )
        f = open('data/bones_length_hall6.json', )
        gt_bones_lens = json.load(f)

        """bones_means_ = []
        for i in gt_bones_lens.keys():
            bones_means_.append(gt_bones_lens[i])
        bones_means = torch.from_numpy(np.mean(np.array(bones_means_), axis=(0, 1))).cuda()
        print("bones_means : " + str(bones_means))
        bones = torch.from_numpy(np.array(cfg.HALL6_DATA.BONES)).cuda()"""
        bone_names_h36m = cfg.HALL6_DATA.BONES_NAMES
        bones_h36m = cfg.HALL6_DATA.BONES
        bones_means_ = []
        bones = []
        count_subj=0
        symmetry_bones = [[],[]]
        bone_names_in_bones_list = []
        for sub_id in gt_bones_lens.keys():
            sub_processed = gt_bones_lens[sub_id]['h36m']
            bone_id = 0
            bones_means_.append([])
            bone_id_in_bones_list = 0
            for bone_name in bone_names_h36m:
                if bone_name in ['Head', 'LeftShoulder', 'RightShoulder']:
                    bone_id += 1
                    continue
                if bone_name in sub_processed.keys():
                    bones_means_[-1].append(sub_processed[bone_name]/100)
                    if count_subj == 0:
                        bones.append(bones_h36m[bone_id])
                        bone_names_in_bones_list.append(bone_name)
                        if bone_id in cfg.HALL6_DATA.BONES_SYMMETRY[0]:
                            symmetry_bones[0].append(bone_id_in_bones_list)
                        if bone_id in cfg.HALL6_DATA.BONES_SYMMETRY[1]:
                            symmetry_bones[1].append(bone_id_in_bones_list)
                    bone_id_in_bones_list += 1
                bone_id +=1
            count_subj +=1
        cfg.HALL6_DATA.BONES_SYMMETRY = symmetry_bones
        print("bone_names_in_bones_list : " + str(bone_names_in_bones_list))
        print("cfg.HALL6_DATA.BONES_SYMMETRY : " + str(cfg.HALL6_DATA.BONES_SYMMETRY))
        torch.from_numpy(np.array(bones_means_)).cuda()
        bones_means = torch.from_numpy(np.mean(np.array(bones_means_), axis=(0))).cuda()
        bones = torch.from_numpy(np.array(bones)).cuda()
        print("bones_means : " + str(bones_means))
        print("bones : " + str(bones))
        # f = open('bone_priors_std.json', )
        # data = json.load(f)
        # bones_stds = []
        # for i in data.keys():
        #     bones_stds.append(data[i])
        bones_stds = torch.from_numpy(np.std(np.array(bones_means_), axis=(0))).cuda()
        print("bones_stds : " + str(bones_stds))
        """for i in gt_bones_lens.keys():
            gt_bones_lens[i] = torch.from_numpy(np.array(gt_bones_lens[i]))"""
        #########
        torch.autograd.set_detect_anomaly(True)
        while epoch < num_hyperparam_epochs:
            start_time = time.time()
            model.train()
            process = tqdm(total=train_generator.num_batches)
            idx = 0
            for batch_2d, sub_action, batch_flip in train_generator.next_epoch():

                if idx >=100:
                    break
                if EVAL:
                    break
                idx += 1
                process.update(1)
                inputs = torch.from_numpy(batch_2d.astype('float32'))
                assert inputs.shape[-2] == 8  # (p2d_gt, p2d_pre, p3d, vis)
                inputs_2d_gt = inputs[..., :, :2, :]
                inputs_2d_pre = inputs[..., 2:4, :]
                cam_3d = inputs[..., 4:7, :]
                B, T, V, C, N = cam_3d.shape
                if use_2d_gt:
                    vis = torch.ones(B, T, V, 1, N)
                else:
                    vis = inputs[..., 7:8, :]

                if cfg.TRAIN.NUM_AUGMENT_VIEWS:
                    vis = torch.cat((vis, torch.ones(B, T, V, 1, cfg.TRAIN.NUM_AUGMENT_VIEWS)), dim=-1)

                inputs_3d_gt = cam_3d.cuda()
                view_list = cfg.HALL6_DATA.TRAIN_CAMERAS
                N = len(view_list)
                if cfg.TRAIN.NUM_AUGMENT_VIEWS > 0:
                    pos_gt_3d_tmp = copy.deepcopy(inputs_3d_gt)
                    pos_gt_2d, pos_gt_3d = data_aug(pos_gt_2d=inputs_2d_gt, pos_gt_3d=pos_gt_3d_tmp)
                    pos_pre_2d = torch.cat((inputs_2d_pre, pos_gt_2d[..., inputs_2d_pre.shape[-1]:]), dim=-1)

                    if use_2d_gt:
                        h36_inp = pos_gt_2d[..., view_list]
                    else:
                        h36_inp = pos_pre_2d[..., view_list]
                    pos_gt = pos_gt_3d[..., view_list]

                else:
                    if use_2d_gt:
                        h36_inp = inputs_2d_gt[..., view_list]
                    else:
                        h36_inp = inputs_2d_pre[..., view_list]
                    pos_gt = inputs_3d_gt[..., view_list]
                p3d_gt_ori = copy.deepcopy(pos_gt)
                p3d_root = copy.deepcopy(pos_gt[:, :, :1])  # (B,T, 1, 3, N)
                pos_gt = pos_gt - pos_gt[:, :, :1]
                p3d_gt_abs = pos_gt + p3d_root
                optimizer.zero_grad()
                inp = torch.cat((h36_inp, vis[..., view_list]), dim=-2)
                if cfg.NETWORK.USE_GT_TRANSFORM or cfg.TRAIN.USE_ROT_LOSS:
                    rotation = get_rotation(pos_gt[:, :1])
                else:
                    rotation = None
                if cfg.TRAIN.LEARN_CAM_PARM:
                    params = model(inp, rotation)
                else:
                    if cfg.TRAIN.USE_INTER_LOSS:
                        print('Input shape is {}'.format(inp.shape))
                        if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
                            out, out_full, other_out, tran, pred_rot = model(inp, rotation)
                        else:
                            out, other_out, tran, pred_rot = model(inp, rotation)  # mask:(B, 1, 1, 1, N, N)
                    else:
                        out = model(inp, rotation)
                if cfg.TRAIN.PREDICT_REDUCED_PARAMETERS:
                    extri, extri_inv, proj = HumanCam.recover_extri_extri_inv_predicted_params(params, sub_action, reduced_params=True, view_list=view_list)
                else:
                    extri, extri_inv, proj = HumanCam.recover_extri_extri_inv_predicted_params(params, sub_action, view_list=view_list)
                if cfg.TRAIN.PREDICT_ROOT:
                    prj_2dpre_to_3d, stats_sing_values, trj_w3d = HumanCam.p2d_cam3d_batch_with_root(
                        h36_inp[:, pad:pad + 1, :, :, :].to(params.device), sub_action, view_list, debug=False,
                        extri=extri, proj=proj, confidences=vis[:, pad:pad+1, :, :, view_list])
                    # prj_2dpre_to_3d, stats_sing_values, trj_w3d = HumanCam.p2d_cam3d_batch_with_root(
                    #     h36_inp[:, pad:pad + 1, :, :, :].to(params.device), sub_action, view_list, debug=False,
                    #     extri=extri, proj=proj)
                else:
                    prj_2dpre_to_3d, stats_sing_values, trj_w3d = HumanCam.p2d_cam3d_batch(
                        h36_inp[:, pad:pad + 1, :, :, :].to(params.device), sub_action, view_list, debug=False,
                        extri=extri, proj=proj, confidences=vis[:, pad:pad+1, :, :, view_list])
                if not cfg.TRAIN.PREDICT_ROOT:
                    prj_2dpre_to_3d = torch.maximum(
                        torch.minimum(prj_2dpre_to_3d, torch.ones_like(prj_2dpre_to_3d) * 1.5),
                        torch.ones_like(prj_2dpre_to_3d) * (-1.5))
                    prj_2dpre_to_3d = prj_2dpre_to_3d + p3d_root[:, pad:pad + 1]
                else:
                    root_pred = prj_2dpre_to_3d[:, :, :1]
                    root_pred_clipped = torch.maximum(torch.minimum(root_pred, torch.ones_like(root_pred) * 20),
                                                      torch.ones_like(root_pred) * (-20))
                    prj_2dpre_to_3d = torch.maximum(
                        torch.minimum(prj_2dpre_to_3d - root_pred, torch.ones_like(prj_2dpre_to_3d) * 1.5),
                        torch.ones_like(prj_2dpre_to_3d) * (-1.5)) + root_pred_clipped
                pose_2D_from3D = HumanCam.p3d_im2d_batch(prj_2dpre_to_3d, sub_action, view_list, with_distor=True,
                                                         flip=batch_flip)
                pose_2D_from3D = pose_2D_from3D.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, N, J. C)
                inputs_2d_gt = h36_inp.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, N, J. C)
                if idx%300 == 0:
                    for v in range(4):
                        print(inputs_2d_gt.shape)
                        h_inp_2d = inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, v, :, 0].detach().numpy()
                        w_inp_2d = inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, v, :, 1].detach().numpy()
                        print(pose_2D_from3D.permute(0, 2, 3, 4, 1).detach().cpu().numpy().shape)
                        h_3D_to_2D = pose_2D_from3D.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, v, :, 0, 0]
                        w_3D_to_2D = pose_2D_from3D.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, v, :, 1, 0]
                        plt.figure("view : "+str(v))
                        plt.scatter(h_inp_2d, w_inp_2d, color='r', marker='o', label="input_2d")
                        plt.scatter(h_3D_to_2D, w_3D_to_2D, color='g', marker='+', label="prj_3dgt_abs_to_2d")
                        plt.legend()
                        fig = plt.figure("view " + str(v))
                        ax = fig.add_subplot(projection='3d')

                        print("prj_2dpre_to_3d.shape : "+str(prj_2dpre_to_3d.shape))
                        print("p3d_gt_abs.shape : "+str(p3d_gt_abs.shape))
                        x_out = prj_2dpre_to_3d[0, 0, :, 0, v].detach().cpu().numpy()
                        y_out = prj_2dpre_to_3d[0, 0, :, 1, v].detach().cpu().numpy()
                        z_out = prj_2dpre_to_3d[0, 0, :, 2, v].detach().cpu().numpy()

                        x_gt = p3d_gt_abs[0, 0, :, 0, v].detach().cpu().numpy()
                        y_gt = p3d_gt_abs[0, 0, :, 1, v].detach().cpu().numpy()
                        z_gt = p3d_gt_abs[0, 0, :, 2, v].detach().cpu().numpy()
                        ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
                        ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
                        ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
                        plt.savefig("view " + str(v))
                    plt.show()

                if cfg.TRAIN.USE_BONES_3D_VIEWS_CONSIST:
                    bones_3d_pred = torch.unsqueeze(get_batch_bones_lens(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3), bones),
                                                    -1)
                    bones_3d_pred = torch.unsqueeze(bones_3d_pred, 3).repeat(1, 1, 1, N, 1, 1)  # (B, T, N, N, J, C)
                    bones_3d_pred_permuted = bones_3d_pred.clone().permute(0, 1, 3, 2, 4, 5)
                    bones_views_consist_3D = torch.mean(torch.square(bones_3d_pred - bones_3d_pred_permuted.detach()))
                if cfg.TRAIN.USE_LOSS_BONES_2D:
                    bones_2d_pred = torch.unsqueeze(get_batch_bones_lens(pose_2D_from3D, bones), -1)
                    bones_2d_inp = torch.unsqueeze(
                        get_batch_bones_lens(inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device), bones), -1)
                    loss_bones_2d = mpjpe(bones_2d_pred, bones_2d_inp)
                    print("loss_bones_2d : " + str(loss_bones_2d))

                if cfg.TRAIN.USE_BONE_DIR_VECT:
                    bones_2d_pred_direction_vect = get_batch_bones_directions(pose_2D_from3D, bones)
                    bones_2d_inp_direction_vect = get_batch_bones_directions(
                        inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device), bones)
                    loss_direct_vect_bones_2d = bones_2d_pred_direction_vect - bones_2d_inp_direction_vect
                    loss_direct_vect_bones_2d = torch.norm(torch.unsqueeze(loss_direct_vect_bones_2d, -1), dim=-1)
                    loss_direct_vect_bones_2d = torch.mean(loss_direct_vect_bones_2d)
                    summary_writer.add_scalar("loss_direct_vect_bones_2d/iter", loss_direct_vect_bones_2d, iters)
                    print("loss_direct_vect_bones_2d : " + str(loss_direct_vect_bones_2d))
                if cfg.TRAIN.USE_BODY_DIR_VECT:
                    body_2d_pred_direction_vect = get_batch_body_directions(pose_2D_from3D)
                    body_2d_inp_direction_vect = get_batch_body_directions(
                        inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device))
                    loss_direct_vect_body_2d = body_2d_pred_direction_vect - body_2d_inp_direction_vect
                    loss_direct_vect_body_2d = torch.norm(torch.unsqueeze(loss_direct_vect_body_2d, -1), dim=-1)
                    loss_direct_vect_body_2d = torch.mean(loss_direct_vect_body_2d)
                    summary_writer.add_scalar("loss_direct_vect_body_2d/iter", loss_direct_vect_body_2d, iters)
                    print("loss_direct_vect_body_2d : " + str(loss_direct_vect_body_2d))

                if cfg.TRAIN.USE_SYM_LOSS:
                    l_bones = torch.from_numpy(np.array(cfg.HALL6_DATA.BONES_SYMMETRY)[0]).to(params.device)
                    r_bones = torch.from_numpy(np.array(cfg.HALL6_DATA.BONES_SYMMETRY)[1]).to(params.device)
                    sym_loss = symetry_loss(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones,
                                            l_bones, r_bones)
                    summary_writer.add_scalar("sym_loss/iter", sym_loss, iters)
                loss_2d = mpjpe_per_view(pose_2D_from3D, inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device))
                print("loss_2d : " + str(loss_2d))
                print("prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:,:,:]" + str(
                    prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :].shape))

                pred_bone_mean, pred_bone_std = bone_losses(
                    prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action,
                    cfg=cfg)  # pose_3D_in_cam_space, p3d_gt_abs[:, pad:pad + 1]
                _, pred_bone_std_per_view = bone_losses(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :],
                                                        bones, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg,
                                                        get_per_view_values=True)  # pose_3D_in_cam_space, p3d_gt_abs[:, pad:pad + 1]
                bone_std_loss_per_view = torch.mean(pred_bone_std_per_view, dim=(0, 2))
                bone_std_loss = torch.mean(pred_bone_std)
                bone_prior_loss = get_bone_prior_loss(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :],
                                                      bones, bones_means, bones_stds)
                if cfg.TRAIN.USE_GT_BONES_LENS:
                    per_participant_bone_len_loss = bone_len_loss(gt_bones_lens,
                                                                  prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:,
                                                                  :, :], bones,cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg,
                                                                  std_bones_len_prior=bones_stds)
                    summary_writer.add_scalar("per_participant_bone_len_loss/iter", per_participant_bone_len_loss,
                                              iters)
                print("bone_prior_loss : " + str(bone_prior_loss))
                ########################## metrics ##########################
                search = False
                if cfg.TRAIN.PREDICT_ROOT and (not search):
                    prj_2dpre_to_3d = prj_2dpre_to_3d - root_pred_clipped + p3d_root[:, pad:pad + 1]
                    loss_copy = mpjpe_per_view(prj_2dpre_to_3d.permute(0, 1, 2, 4, 3).contiguous(),
                                               p3d_gt_abs[:, pad:pad + 1].permute(0, 1, 2, 4, 3).contiguous())
                    for mpjpe_views in range(N):
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_gt_root_view_{}/iter".format(mpjpe_views),
                                                      loss_copy[mpjpe_views], iters)
                    if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                        summary_writer.add_scalar("loss_copy_gt_root/iter", torch.mean(loss_copy),
                                                  iters)  # loss = loss_copy if cfg.TRAIN.UNSUPERVISE==F$
                    if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                        summary_writer.add_scalar("loss_copy_gt_root_best_std_view/iter",
                                                  loss_copy[torch.argmin(bone_std_loss_per_view)],
                                                  iters)  # loss = loss_copy if cfg.TRAIN.UNSUPERVISE==F$
                    predicted_ = prj_2dpre_to_3d.permute(0, 1, 2, 4, 3).contiguous().view(B * 1 * N, V,
                                                                                          C).cpu().detach()
                    target_ = p3d_gt_abs[:, pad:pad + 1].permute(0, 1, 2, 4, 3).contiguous().view(B * 1 * N, V,
                                                                                                  C).cpu().detach()
                    try:
                        loss_copy = p_mpjpe_per_view(cfg, predicted_, target_, n_views=N)
                    except:
                        loss_copy=[-1 for i in range(N)]
                    for mpjpe_views in range(N):
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_gt_root_p_mpjpe_view_{}/iter".format(mpjpe_views),
                                                      loss_copy[mpjpe_views], iters)
                    if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                        summary_writer.add_scalar("loss_copy_gt_root_p_mpjpe/iter", np.mean(loss_copy), iters)
                    if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                        summary_writer.add_scalar("loss_copy_gt_root_p_mpjpe_best_std_view/iter",
                                                  loss_copy[torch.argmin(bone_std_loss_per_view)], iters)
                elif (not cfg.TRAIN.PREDICT_ROOT) and (not search):
                    loss_copy = mpjpe_per_view(prj_2dpre_to_3d.permute(0, 1, 2, 4, 3).contiguous(),
                                               p3d_gt_abs[:, pad:pad + 1].permute(0, 1, 2, 4, 3).contiguous())
                    for mpjpe_views in range(N):
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_view_{}/iter".format(mpjpe_views),
                                                      loss_copy[mpjpe_views], iters)
                    if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                        summary_writer.add_scalar("loss_copy/iter", torch.mean(loss_copy),
                                                  iters)  # loss = loss_copy if cfg.TRAIN.UNSUPERVISE==F$
                    if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                        summary_writer.add_scalar("loss_copy_best_std_view/iter",
                                                  loss_copy[torch.argmin(bone_std_loss_per_view)],
                                                  iters)  # loss = loss_copy if cfg.TRAIN.UNSUPERVISE==F$

                    predicted_ = prj_2dpre_to_3d.permute(0, 1, 2, 4, 3).contiguous().view(B * 1 * N, V,
                                                                                          C).cpu().detach()
                    target_ = p3d_gt_abs[:, pad:pad + 1].permute(0, 1, 2, 4, 3).contiguous().view(B * 1 * N, V,
                                                                                                  C).cpu().detach()
                    try:
                        loss_copy = p_mpjpe_per_view(cfg, predicted_, target_)
                        print("p_mpjpe loss_copy: " + str(loss_copy))
                        for mpjpe_views in range(N):
                            if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                                summary_writer.add_scalar("loss_copy_p_mpjpe_view_{}/iter".format(mpjpe_views),
                                                          loss_copy[mpjpe_views], iters)
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_p_mpjpe/iter", np.mean(loss_copy), iters)
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_best_std_view/iter",
                                                      loss_copy[torch.argmin(bone_std_loss_per_view)], iters)
                    except:
                        for mpjpe_views in range(N):
                            if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                                summary_writer.add_scalar("loss_copy_p_mpjpe_view_{}/iter".format(mpjpe_views),
                                                          0, iters)
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_p_mpjpe/iter", 0, iters)
                        if summary_writer is not None and cfg.TRAIN.UNSUPERVISE == True:
                            summary_writer.add_scalar("loss_copy_best_std_view/iter",
                                                      0, iters)
                loss = 0
                unweighted_loss = 0
                if cfg.TRAIN.USE_STD_LOSS:
                    for view_std in range(N):
                        summary_writer.add_scalar("bone_std_loss_view_{}/iter".format(view_std),
                                                  bone_std_loss_per_view[view_std], iters)
                    summary_writer.add_scalar("bone_std_loss/iter", torch.mean(bone_std_loss), iters)
                    loss += std_bone_len_weight * torch.mean(bone_std_loss)
                    print("bone_std_loss : " + str(torch.mean(bone_std_loss)))
                if cfg.TRAIN.USE_2D_LOSS:
                    for view_loss_copy in range(N):
                        summary_writer.add_scalar("loss_2d_view_{}/iter".format(view_loss_copy),
                                                  loss_2d[view_loss_copy], iters)
                    summary_writer.add_scalar("loss_2d/iter", torch.mean(loss_2d), iters)
                    loss += joints_2d_weight*torch.mean(loss_2d)
                if cfg.TRAIN.REGULARIZE_PARAMS:
                    params_reg_loss = params_regularization(params, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg)
                    loss += reg_params_weight * params_reg_loss
                    if summary_writer is not None:
                        summary_writer.add_scalar("params_reg_loss/iter", params_reg_loss, iters)
                # if epoch == 0:
                if cfg.TRAIN.USE_LOSS_BONES_2D:
                    summary_writer.add_scalar("loss_bones_2d/iter", loss_bones_2d, iters)
                    loss += len_2d_bone_weight*loss_bones_2d
                if cfg.TRAIN.USE_SYM_LOSS:
                    loss += symmetry_weight*sym_loss

                if cfg.TRAIN.USE_BONE_DIR_VECT:
                    loss += direction_2d_bone_weight*loss_direct_vect_bones_2d
                if cfg.TRAIN.USE_BODY_DIR_VECT:
                    loss += loss_direct_vect_body_2d
                if cfg.TRAIN.USE_BONES_PRIOR:
                    for view_bone_prior in range(N):
                        summary_writer.add_scalar("bone_prior_loss_view_{}/iter".format(view_bone_prior),
                                                  bone_prior_loss[view_bone_prior], iters)
                    # if epoch==0:
                    summary_writer.add_scalar("bone_prior_loss/iter", torch.mean(bone_prior_loss), iters)
                    loss += human_prior_weight * torch.mean(bone_prior_loss)
                if cfg.TRAIN.USE_GT_BONES_LENS:
                    loss += 0.2 * per_participant_bone_len_loss
                if cfg.TRAIN.USE_BONES_3D_VIEWS_CONSIST:
                    summary_writer.add_scalar("bones_views_consist_3D/iter", bones_views_consist_3D, iters)
                    loss += 1 * bones_views_consist_3D
                if summary_writer is not None:
                    summary_writer.add_scalar("loss_final/iter", loss, iters)
                loss_total = loss
                print('Unsupervised Loss is {}'.format(loss_total))
                loss_total.backward()

                optimizer.step()
                iters += 1
            #free gpu memory
            del loss_total
            del loss
            del loss_2d
            #del loss_bones_2d
            #del loss_direct_vect_bones_2d
            #del loss_direct_vect_body_2d
            del bone_prior_loss
            #del bone_std_loss
            #del bone_std_loss_per_view
            #del sym_loss
            del inputs



            #eval
            unweighted_losses = []
            weighted_losses = []
            losses_3D = []
            idx = 0
            load_state(model, model_test)
            model_test.eval()
            for batch_2d, sub_action, batch_flip in test_generator.next_epoch():
                if idx >=100:
                    break
                idx += 1
                if EVAL:
                    break
                #process.update(1)
                inputs = torch.from_numpy(batch_2d.astype('float32'))
                assert inputs.shape[-2] == 8  # (p2d_gt, p2d_pre, p3d, vis)
                inputs_2d_gt = inputs[..., :, :2, :]
                inputs_2d_pre = inputs[..., 2:4, :]
                cam_3d = inputs[..., 4:7, :]
                B, T, V, C, N = cam_3d.shape
                if use_2d_gt:
                    vis = torch.ones(B, T, V, 1, N)
                else:
                    vis = inputs[..., 7:8, :]

                if cfg.TRAIN.NUM_AUGMENT_VIEWS:
                    vis = torch.cat((vis, torch.ones(B, T, V, 1, cfg.TRAIN.NUM_AUGMENT_VIEWS)), dim=-1)

                inputs_3d_gt = cam_3d.cuda()
                view_list = cfg.HALL6_DATA.TRAIN_CAMERAS
                N = len(view_list)
                if cfg.TRAIN.NUM_AUGMENT_VIEWS > 0:
                    pos_gt_3d_tmp = copy.deepcopy(inputs_3d_gt)
                    pos_gt_2d, pos_gt_3d = data_aug(pos_gt_2d=inputs_2d_gt, pos_gt_3d=pos_gt_3d_tmp)
                    pos_pre_2d = torch.cat((inputs_2d_pre, pos_gt_2d[..., inputs_2d_pre.shape[-1]:]), dim=-1)

                    if use_2d_gt:
                        h36_inp = pos_gt_2d[..., view_list]
                    else:
                        h36_inp = pos_pre_2d[..., view_list]
                    pos_gt = pos_gt_3d[..., view_list]

                else:
                    if use_2d_gt:
                        h36_inp = inputs_2d_gt[..., view_list]
                    else:
                        h36_inp = inputs_2d_pre[..., view_list]
                    pos_gt = inputs_3d_gt[..., view_list]
                p3d_gt_ori = copy.deepcopy(pos_gt)
                p3d_root = copy.deepcopy(pos_gt[:, :, :1])  # (B,T, 1, 3, N)
                pos_gt = pos_gt - pos_gt[:, :, :1]
                p3d_gt_abs = pos_gt + p3d_root
                optimizer.zero_grad()
                inp = torch.cat((h36_inp, vis[..., view_list]), dim=-2)
                if cfg.NETWORK.USE_GT_TRANSFORM or cfg.TRAIN.USE_ROT_LOSS:
                    rotation = get_rotation(pos_gt[:, :1])
                else:
                    rotation = None
                if cfg.TRAIN.LEARN_CAM_PARM:
                    params = model_test(inp, rotation)
                else:
                    if cfg.TRAIN.USE_INTER_LOSS:
                        print('Input shape is {}'.format(inp.shape))
                        if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
                            out, out_full, other_out, tran, pred_rot = model_test(inp, rotation)
                        else:
                            out, other_out, tran, pred_rot = model_test(inp, rotation)  # mask:(B, 1, 1, 1, N, N)
                    else:
                        out = model_test(inp, rotation)
                if cfg.TRAIN.PREDICT_REDUCED_PARAMETERS:
                    extri, extri_inv, proj = HumanCam.recover_extri_extri_inv_predicted_params(params, sub_action, reduced_params=True, view_list=view_list)
                else:
                    extri, extri_inv, proj = HumanCam.recover_extri_extri_inv_predicted_params(params, sub_action, view_list=view_list)
                if cfg.TRAIN.PREDICT_ROOT:
                    prj_2dpre_to_3d, stats_sing_values, trj_w3d = HumanCam.p2d_cam3d_batch_with_root(
                        h36_inp[:, pad:pad + 1, :, :, :].to(params.device), sub_action, view_list, debug=False,
                        extri=extri, proj=proj, confidences=vis[:, pad:pad+1, :, :, view_list])
                    # prj_2dpre_to_3d, stats_sing_values, trj_w3d = HumanCam.p2d_cam3d_batch_with_root(
                    #     h36_inp[:, pad:pad + 1, :, :, :].to(params.device), sub_action, view_list, debug=False,
                    #     extri=extri, proj=proj)
                else:
                    prj_2dpre_to_3d, stats_sing_values, trj_w3d = HumanCam.p2d_cam3d_batch(
                        h36_inp[:, pad:pad + 1, :, :, :].to(params.device), sub_action, view_list, debug=False,
                        extri=extri, proj=proj, confidences=vis[:, pad:pad+1, :, :, view_list])
                if not cfg.TRAIN.PREDICT_ROOT:
                    prj_2dpre_to_3d = torch.maximum(
                        torch.minimum(prj_2dpre_to_3d, torch.ones_like(prj_2dpre_to_3d) * 1.5),
                        torch.ones_like(prj_2dpre_to_3d) * (-1.5))
                    prj_2dpre_to_3d = prj_2dpre_to_3d + p3d_root[:, pad:pad + 1]
                else:
                    root_pred = prj_2dpre_to_3d[:, :, :1]
                    root_pred_clipped = torch.maximum(torch.minimum(root_pred, torch.ones_like(root_pred) * 20), torch.ones_like(root_pred) * (-20))
                    prj_2dpre_to_3d = torch.maximum(
                        torch.minimum(prj_2dpre_to_3d - root_pred, torch.ones_like(prj_2dpre_to_3d) * 1.5),
                        torch.ones_like(prj_2dpre_to_3d) * (-1.5)) + root_pred_clipped
                pose_2D_from3D = HumanCam.p3d_im2d_batch(prj_2dpre_to_3d, sub_action, view_list, with_distor=True,
                                                         flip=batch_flip)
                pose_2D_from3D = pose_2D_from3D.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, N, J. C)
                inputs_2d_gt = h36_inp.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, N, J. C)
                if idx%300 == 0:
                    for v in range(4):
                        print(inputs_2d_gt.shape)
                        h_inp_2d = inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, v, :, 0].detach().numpy()
                        w_inp_2d = inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, v, :, 1].detach().numpy()
                        print(pose_2D_from3D.permute(0, 2, 3, 4, 1).detach().cpu().numpy().shape)
                        h_3D_to_2D = pose_2D_from3D.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, v, :, 0, 0]
                        w_3D_to_2D = pose_2D_from3D.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, v, :, 1, 0]
                        plt.figure("view : "+str(v))
                        plt.scatter(h_inp_2d, w_inp_2d, color='r', marker='o', label="input_2d")
                        plt.scatter(h_3D_to_2D, w_3D_to_2D, color='g', marker='+', label="prj_3dgt_abs_to_2d")
                        plt.legend()
                        fig = plt.figure("view " + str(v))
                        ax = fig.add_subplot(projection='3d')

                        print("prj_2dpre_to_3d.shape : "+str(prj_2dpre_to_3d.shape))
                        print("p3d_gt_abs.shape : "+str(p3d_gt_abs.shape))
                        x_out = prj_2dpre_to_3d[0, 0, :, 0, v].detach().cpu().numpy()
                        y_out = prj_2dpre_to_3d[0, 0, :, 1, v].detach().cpu().numpy()
                        z_out = prj_2dpre_to_3d[0, 0, :, 2, v].detach().cpu().numpy()

                        x_gt = p3d_gt_abs[0, 0, :, 0, v].detach().cpu().numpy()
                        y_gt = p3d_gt_abs[0, 0, :, 1, v].detach().cpu().numpy()
                        z_gt = p3d_gt_abs[0, 0, :, 2, v].detach().cpu().numpy()
                        ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
                        ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
                        ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
                        plt.savefig("view " + str(v))
                    plt.show()

                if cfg.TRAIN.USE_BONES_3D_VIEWS_CONSIST:
                    bones_3d_pred = torch.unsqueeze(get_batch_bones_lens(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3), bones),
                                                    -1)
                    bones_3d_pred = torch.unsqueeze(bones_3d_pred, 3).repeat(1, 1, 1, N, 1, 1)  # (B, T, N, N, J, C)
                    bones_3d_pred_permuted = bones_3d_pred.clone().permute(0, 1, 3, 2, 4, 5)
                    bones_views_consist_3D = torch.mean(torch.square(bones_3d_pred - bones_3d_pred_permuted.detach()))
                if cfg.TRAIN.USE_LOSS_BONES_2D:
                    bones_2d_pred = torch.unsqueeze(get_batch_bones_lens(pose_2D_from3D, bones), -1)
                    bones_2d_inp = torch.unsqueeze(
                        get_batch_bones_lens(inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device), bones), -1)
                    loss_bones_2d = mpjpe(bones_2d_pred, bones_2d_inp)
                    print("loss_bones_2d : " + str(loss_bones_2d))

                if cfg.TRAIN.USE_BONE_DIR_VECT:
                    bones_2d_pred_direction_vect = get_batch_bones_directions(pose_2D_from3D, bones)
                    bones_2d_inp_direction_vect = get_batch_bones_directions(
                        inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device), bones)
                    loss_direct_vect_bones_2d = bones_2d_pred_direction_vect - bones_2d_inp_direction_vect
                    loss_direct_vect_bones_2d = torch.norm(torch.unsqueeze(loss_direct_vect_bones_2d, -1), dim=-1)
                    loss_direct_vect_bones_2d = torch.mean(loss_direct_vect_bones_2d)
                    summary_writer.add_scalar("loss_direct_vect_bones_2d/iter", loss_direct_vect_bones_2d, iters)
                    print("loss_direct_vect_bones_2d : " + str(loss_direct_vect_bones_2d))
                if cfg.TRAIN.USE_BODY_DIR_VECT:
                    body_2d_pred_direction_vect = get_batch_body_directions(pose_2D_from3D)
                    body_2d_inp_direction_vect = get_batch_body_directions(
                        inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device))
                    loss_direct_vect_body_2d = body_2d_pred_direction_vect - body_2d_inp_direction_vect
                    loss_direct_vect_body_2d = torch.norm(torch.unsqueeze(loss_direct_vect_body_2d, -1), dim=-1)
                    loss_direct_vect_body_2d = torch.mean(loss_direct_vect_body_2d)
                    summary_writer.add_scalar("loss_direct_vect_body_2d/iter", loss_direct_vect_body_2d, iters)
                    print("loss_direct_vect_body_2d : " + str(loss_direct_vect_body_2d))

                if cfg.TRAIN.USE_SYM_LOSS:
                    l_bones = torch.from_numpy(np.array(cfg.HALL6_DATA.BONES_SYMMETRY)[0]).to(params.device)
                    r_bones = torch.from_numpy(np.array(cfg.HALL6_DATA.BONES_SYMMETRY)[1]).to(params.device)
                    sym_loss = symetry_loss(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones,
                                            l_bones, r_bones)
                    summary_writer.add_scalar("sym_loss/iter", sym_loss, iters)
                loss_2d = mpjpe_per_view(pose_2D_from3D, inputs_2d_gt[:, pad:pad + 1, :, :].to(params.device))
                print("loss_2d : " + str(loss_2d))
                print("prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:,:,:]" + str(
                    prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :].shape))

                pred_bone_mean, pred_bone_std = bone_losses(
                    prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action,
                    cfg=cfg)  # pose_3D_in_cam_space, p3d_gt_abs[:, pad:pad + 1]
                _, pred_bone_std_per_view = bone_losses(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :],
                                                        bones, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg,
                                                        get_per_view_values=True)  # pose_3D_in_cam_space, p3d_gt_abs[:, pad:pad + 1]
                bone_std_loss_per_view = torch.mean(pred_bone_std_per_view, dim=(0, 2))
                bone_std_loss = torch.mean(pred_bone_std)
                bone_prior_loss = get_bone_prior_loss(prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:, :, :],
                                                      bones, bones_means, bones_stds)
                if cfg.TRAIN.USE_GT_BONES_LENS:
                    per_participant_bone_len_loss = bone_len_loss(gt_bones_lens,
                                                                  prj_2dpre_to_3d.permute(0, 1, 4, 2, 3).contiguous()[:,
                                                                  :, :], bones,cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg,
                                                                  std_bones_len_prior=bones_stds)
                    summary_writer.add_scalar("per_participant_bone_len_loss/iter", per_participant_bone_len_loss,
                                              iters)
                print("bone_prior_loss : " + str(bone_prior_loss))
                if cfg.TRAIN.PREDICT_ROOT and (not search):
                    # prj_2dpre_to_3d = prj_2dpre_to_3d - prj_2dpre_to_3d[:,:,:1] + p3d_root[:, pad:pad + 1]
                    prj_2dpre_to_3d = prj_2dpre_to_3d - root_pred_clipped + p3d_root[:, pad:pad + 1]
                    loss_copy = mpjpe_per_view(prj_2dpre_to_3d.permute(0, 1, 2, 4, 3).contiguous(),
                                               p3d_gt_abs[:, pad:pad + 1].permute(0, 1, 2, 4, 3).contiguous())
                    losses_3D.append(torch.mean(loss_copy).detach().cpu().numpy())
                loss = 0
                unweighted_loss = 0
                if cfg.TRAIN.USE_STD_LOSS:
                    for view_std in range(N):
                        summary_writer.add_scalar("bone_std_loss_view_{}/iter".format(view_std),
                                                  bone_std_loss_per_view[view_std], iters)
                    summary_writer.add_scalar("bone_std_loss/iter", torch.mean(bone_std_loss), iters)
                    loss += std_bone_len_weight * torch.mean(bone_std_loss)
                    unweighted_loss += torch.mean(bone_std_loss)
                    print("bone_std_loss : " + str(torch.mean(bone_std_loss)))
                if cfg.TRAIN.USE_2D_LOSS:
                    for view_loss_copy in range(N):
                        summary_writer.add_scalar("loss_2d_view_{}/iter".format(view_loss_copy),
                                                  loss_2d[view_loss_copy], iters)
                    summary_writer.add_scalar("loss_2d/iter", torch.mean(loss_2d), iters)
                    loss += joints_2d_weight*torch.mean(loss_2d)
                    unweighted_loss += torch.mean(loss_2d)
                if cfg.TRAIN.REGULARIZE_PARAMS:
                    params_reg_loss = params_regularization(params, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg)
                    loss += reg_params_weight * params_reg_loss
                    unweighted_loss += params_reg_loss
                    if summary_writer is not None:
                        summary_writer.add_scalar("params_reg_loss/iter", params_reg_loss, iters)
                # if epoch == 0:
                if cfg.TRAIN.USE_LOSS_BONES_2D:
                    summary_writer.add_scalar("loss_bones_2d/iter", loss_bones_2d, iters)
                    loss += len_2d_bone_weight*loss_bones_2d
                    unweighted_loss += loss_bones_2d
                if cfg.TRAIN.USE_SYM_LOSS:
                    loss += symmetry_weight*sym_loss
                    unweighted_loss += sym_loss

                if cfg.TRAIN.USE_BONE_DIR_VECT:
                    loss += direction_2d_bone_weight*loss_direct_vect_bones_2d
                    unweighted_loss += loss_direct_vect_bones_2d
                if cfg.TRAIN.USE_BODY_DIR_VECT:
                    loss += loss_direct_vect_body_2d
                    unweighted_loss += loss_direct_vect_body_2d

                if cfg.TRAIN.USE_BONES_PRIOR:
                    for view_bone_prior in range(N):
                        summary_writer.add_scalar("bone_prior_loss_view_{}/iter".format(view_bone_prior),
                                                  bone_prior_loss[view_bone_prior], iters)
                    summary_writer.add_scalar("bone_prior_loss/iter", torch.mean(bone_prior_loss), iters)
                    loss += human_prior_weight * torch.mean(bone_prior_loss)
                    unweighted_loss += torch.mean(bone_prior_loss)
                if cfg.TRAIN.USE_GT_BONES_LENS:
                    loss += 0.2 * per_participant_bone_len_loss
                if cfg.TRAIN.USE_BONES_3D_VIEWS_CONSIST:
                    summary_writer.add_scalar("bones_views_consist_3D/iter", bones_views_consist_3D, iters)
                    loss += 1 * bones_views_consist_3D
                    unweighted_loss += bones_views_consist_3D
                if summary_writer is not None:
                    summary_writer.add_scalar("loss_final/iter", loss, iters)
                unweighted_losses.append(unweighted_loss.detach().cpu().numpy())
                weighted_losses.append(loss.detach().cpu().numpy())
            process.close()

            summary_writer.add_scalar("unweighted_loss/epoch", np.mean(np.array(unweighted_losses)), epoch)
            summary_writer.add_scalar("loss/epoch", np.mean(np.array(weighted_losses)), epoch)
            trial.set_user_attr("weighted_losses", str(np.mean(np.array(weighted_losses))))
            trial.set_user_attr("losses_3D", str(np.mean(np.array(losses_3D))))

            epoch += 1

            if np.mean(np.array(unweighted_losses)) < best_result:
                best_result = np.mean(np.array(unweighted_losses))
                best_state_dict = copy.deepcopy(model.module.state_dict())
                best_result_epoch = epoch
            elapsed = (time.time() - start_time) / 60
            # free gpu memory
            del unweighted_loss
            del loss
            del loss_2d
            # del loss_bones_2d
            # del loss_direct_vect_bones_2d
            # del loss_direct_vect_body_2d
            del bone_prior_loss
            # del bone_std_loss
            # del bone_std_loss_per_view
            # del sym_loss
            del inputs
            del inputs_2d_gt
            del inputs_2d_pre
            del cam_3d
            print(
                'epoch:{:3} time:{:.2f} lr:{:.9f} best_result_epoch:{:3} best_result:{:.3f}'.format(epoch, elapsed, lr,
                                                                                                    best_result_epoch,
                                                                                                    best_result))
            print('checkpoint:{}'.format(cfg.TRAIN.CHECKPOINT))
            if epoch < 60:
                # Decay learning rate exponentially
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
            momentum = initial_momentum * np.exp(
                -epoch / cfg.TRAIN.NUM_EPOCHES * np.log(initial_momentum / final_momentum))
            model.module.set_bn_momentum(momentum)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model': model.module.state_dict(),
                'best_epoch': best_result_epoch,
                'best_model': best_state_dict,
            }, cfg.TRAIN.CHECKPOINT)
    return np.mean(np.array(unweighted_losses))

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = args.hyper_storage_path+"/hyperparam_search"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
print(storage_name)


#def launch_opti():
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=24)