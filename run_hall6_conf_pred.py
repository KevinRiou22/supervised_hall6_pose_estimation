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

print('parsing args!')

print('parsed args!')
update_config(args.cfg)  ###config file->cfg
reset_config(cfg, args)  ###arg -> cfg
print(cfg)

print('p2d detector:{}'.format('gt_p2d' if cfg.DATA.USE_GT_2D else cfg.H36M_DATA.P2D_DETECTOR))
HumanCam = Human36mCamera(cfg)

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
my_data_pth = path_dataset+args.data_name+".npz"
#my_data_pth = path_dataset+"triangulated_3D_with_distor_2D_structure.npz"#'hall6.npz' ####one example:t1_o1_ex7
my_data = np.load(my_data_pth, allow_pickle=True)
my_data = dict(my_data)
actions = []
N_frame_action_dict = {}
for sub in operators:
    keypoints['S{}'.format(sub)] = my_data['S{}'.format(sub)].item()
    print("subject {} is processing".format(sub))
    ############ coco to h36m ######################
    keypoints_new['S{}'.format(sub)] = copy.deepcopy(my_data['S{}'.format(sub)].item())

    t_i=0
    for task in tasks:
        for example in examples:
            if sub==4 and example==6:
                if task==1 or task==3:
                    continue
            if 'task{}_example{}'.format(task, example) in keypoints_new['S{}'.format(sub)]:
                print('Processing task {} example {}......'.format(task, example))
                for id, i in enumerate(cfg.HALL6_DATA.TRAIN_CAMERAS):
                    #print('Processing view {}......'.format(i))
                    #print(keypoints_new['S{}'.format(sub)].keys())
                    """keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,0,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,11,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,12,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,5,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,6,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,7,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,0,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,10,:] = (keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,3,:] + keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,4,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,9,:] = (keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,8,:] + keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,10,:])/2
                    keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],:] = keypoints['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i][:,[12, 14, 16, 11, 13, 15, 5, 7, 9, 6, 8, 10],:]"""
                actions.append('task{}_example{}'.format(task, example))
                n_frame_current_ex = keypoints_new['S{}'.format(sub)]['task{}_example{}'.format(task, example)][i].shape[0]
                print('n_frame_current_ex:{}'.format(n_frame_current_ex))
                if 'S{}'.format(sub) in cfg.HALL6_DATA.SUBJECTS_TEST:
                    #print('test')
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

    print('** Starting.')

    data_aug = DataAug(cfg, add_view=cfg.TRAIN.NUM_AUGMENT_VIEWS)
    iters = 0
    msefun = torch.nn.L1Loss()
    num_train_views = len(cfg.HALL6_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
    last_loss = 0
    #########
    import json

    #f = open('bone_priors_mean.json', )
    f = open('data/bones_length_hall6_2d_pose_structure.json', )
    bones_prior_dict = json.load(f)
    bone_names_h36m = cfg.HALL6_DATA.BONES_NAMES
    bones_h36m = cfg.HALL6_DATA.BONES
    bones_means_ = []
    bones = []
    count_subj=0
    symmetry_bones = [[],[]]
    bone_names_in_bones_list = []
    gt_bones_lens={}
    processed_lens = torch.from_numpy(np.load("data/bones_length_hall6_triang_measure.npy"))

    for sub_id in bones_prior_dict.keys():
        sub_gt_bones_lens = []
        sub_processed = bones_prior_dict[sub_id]['h36m']
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
            if bone_name in ['r_ear_r_eye', 'l_ear_l_eye', 'r_eye_nose', 'l_eye_nose']:
                bones_means_[-1].append(0)
                if count_subj == 0:
                    bones.append(bones_h36m[bone_id])
                    bone_names_in_bones_list.append(bone_name)
                    if bone_id in cfg.HALL6_DATA.BONES_SYMMETRY[0]:
                        symmetry_bones[0].append(bone_id_in_bones_list)
                    if bone_id in cfg.HALL6_DATA.BONES_SYMMETRY[1]:
                        symmetry_bones[1].append(bone_id_in_bones_list)
                bone_id_in_bones_list += 1
            bone_id +=1
        gt_bones_lens[sub_id] = processed_lens[count_subj]
        count_subj +=1

    cfg.HALL6_DATA.BONES_SYMMETRY = symmetry_bones
    print("bone_names_in_bones_list : " + str(bone_names_in_bones_list))
    print("cfg.HALL6_DATA.BONES_SYMMETRY : " + str(cfg.HALL6_DATA.BONES_SYMMETRY))
    #gt_bones_lens = torch.from_numpy(np.array(bones_means_)).cuda()
    per_subject_bones = torch.from_numpy(np.array(bones_means_))
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
    #########
    torch.autograd.set_detect_anomaly(True)
    while epoch < cfg.TRAIN.NUM_EPOCHES:
        start_time = time.time()
        model.train()
        process = tqdm(total=train_generator.num_batches)
        idx = 0
        for batch_2d, sub_action, batch_flip in train_generator.next_epoch():
            # if idx > 50:
            #     break
            #start_data_prepare = time.time()
            idx += 1
            if EVAL:
                break
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
            # if cfg.HALL6_DATA.PROJ_Frm_3DCAM == True:
            #    prj_3dgt_to_2d= HumanCam.p3d_im2d(pos_gt, sub_action, view_list)
            p3d_gt_ori = copy.deepcopy(pos_gt)
            p3d_root = copy.deepcopy(pos_gt[:, :, :1])  # (B,T, 1, 3, N)*
            #pos_gt[:, :, :1] = 0
            pos_gt = pos_gt - pos_gt[:, :, :1]

            # print("rel, view 0 : "+str(pos_gt[0,0,:, :, 0]))
            # print("rel, view 1 : "+str(pos_gt[0, 0, :, :, 1]))
            p3d_gt_abs = pos_gt + p3d_root
            # print("abs, view 0 : "+str(p3d_gt_abs[0,0,:, :, 0]))
            # print("abs, view 1 : " + str(p3d_gt_abs[0, 0, :, :, 1]))
            # input()
            optimizer.zero_grad()
            inp = torch.cat((h36_inp, vis[..., view_list]), dim=-2)
            if cfg.NETWORK.USE_GT_TRANSFORM or cfg.TRAIN.USE_ROT_LOSS:
                # 相机之间的旋转
                rotation = get_rotation(pos_gt[:, :1])  # (B, 3, 3, 1, N, N)

                # #相机之间的位移
                # #print(rotation)
                # t = torch.einsum('btjcn,bqcmn->btjqmn', p3d_root[:,:1], rotation[:,:,:,0])#(B, T, 1, 3, N, N)
                # t = t - t[...,:1]
                # t = t.permute(0, 2, 3, 1, 4, 5) #(B, 1, 3, T, N, N)
                # if cfg.NETWORK.M_FORMER.GT_TRANSFORM_MODE == 'rt':
                #     rotation = torch.cat((rotation, t), dim = 1)
            else:
                rotation = None
            #end_data_prepare = time.time()
            #data_prepare_elapsed = (end_data_prepare - start_data_prepare) / 60
            #print('data prepare : time:{:.2f}'.format(data_prepare_elapsed))
            #start_data_prepare = time.time()

            if cfg.TRAIN.USE_INTER_LOSS:
                confs, other_out, tran, pred_rot = model(inp, rotation)  # mask:(B, 1, 1, 1, N, N)
            else:
                confs = model(inp, rotation)
            confs = confs.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, N, J. C)
            #print("vis[:, pad:pad+1, :, :, view_list].shape", vis[:, pad:pad + 1, :, :, view_list].shape)
            #print("confs.shape", confs.shape)

            out, stats_sing_values, _ = HumanCam.p2d_cam3d_batch_with_root(
                h36_inp[:, pad:pad + 1, :, :, :], sub_action, view_list, debug=False,
                confidences=confs[:, pad:pad + 1].permute(0, 1, 3, 4, 2).contiguous())

            # detect nan in out and create a mask

            mask = torch.isnan(out).any(dim=-1).any(dim=-1).any(dim=-1).any(dim=-1)  # (B, T, N)
            #remove nans from out
            out = out[~mask]
            #remove nan from sub_action
            sub_action = np.array(sub_action)[~mask].tolist()

            #loss = mpjpe(out, pos_gt[:, pad:pad + 1])
            loss = bone_len_loss(gt_bones_lens, out.permute(0, 1, 4, 2, 3).contiguous(), bones.to(out.device), cfg.HALL6_DATA.SUBJECTS_TRAIN,batch_subjects=sub_action, cfg=cfg,std_bones_len_prior=bones_stds.to(out.device))
            if summary_writer is not None:
                summary_writer.add_scalar("bone_loss/iter", loss, iters)



            print('Total Loss is {}'.format(loss))
            loss.backward()
            optimizer.step()
            iters += 1
        process.close()

        # for v in range(4):
        #     fig = plt.figure("train view_2d_" + str(v) + "_epoch_" + str(epoch))
        #     print("inputs_2d_gt.shape : " + str(inputs_2d_gt.shape))
        #     h_inp_2d = inp[:, pad:pad + 1, :, :][0, 0, :, 0, v].detach().numpy()
        #     w_inp_2d = inp[:, pad:pad + 1, :, :][0, 0, :, 1, v].detach().numpy()
        #     plt.scatter(h_inp_2d, w_inp_2d, color='r', marker='o', label="input_2d")
        #     fig = plt.figure("train view " + str(v))
        #     ax = fig.add_subplot(projection='3d')
        #
        #     print("pos_gt.shape : "+str(pos_gt.shape))
        #     x_out = out[0, 0, v, :, 0].detach().cpu().numpy()
        #     y_out = out[0, 0, v, :, 1].detach().cpu().numpy()
        #     z_out = out[0, 0, v, :, 2].detach().cpu().numpy()
        #
        #     x_gt = pos_gt[0, 0, v, :, 0].detach().cpu().numpy()
        #     y_gt = pos_gt[0, 0, v, :, 1].detach().cpu().numpy()
        #     z_gt = pos_gt[0, 0, v, :, 2].detach().cpu().numpy()
        #     ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
        #     ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
        #     ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
        #     plt.savefig("view " + str(v))

        ###########eval
        id_eval = 0
        id_traj = 0
        idx_eval=0
        data_npy = {}
        with torch.no_grad():
            if not cfg.TEST.TRIANGULATE:
                load_state(model, model_test)
                model_test.eval()
            NUM_VIEW = len(cfg.HALL6_DATA.TEST_CAMERAS)
            if EVAL:
                eval_start_time = time.time()
            for t_len in cfg.TEST.NUM_FRAMES:
                epoch_loss_valid = 0
                epoch_bone_valid = 0
                action_mpjpe = {}
                action_bone_error = {}
                for act in actions:
                    action_mpjpe[act] = [0] * NUM_VIEW
                    action_bone_error[act] = [0] * NUM_VIEW
                    for i in range(NUM_VIEW):
                        action_mpjpe[act][i] = [0] * (NUM_VIEW + 1)
                        action_bone_error[act][i] = [0] * (NUM_VIEW + 1)
                N = [0] * NUM_VIEW
                for i in range(NUM_VIEW):
                    N[i] = [0] * (NUM_VIEW + 1)

                for num_view in cfg.TEST.NUM_VIEWS:
                    pad_t = t_len // 2
                    for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                        view_list = list(view_list)
                        #view_list =[cfg.HALL6_DATA.TEST_CAMERAS[i] for i in view_list]
                        views_idx = [cfg.HALL6_DATA.TEST_CAMERAS[i] for i in view_list]
                        N[num_view - 1][-1] += 1
                        for i in view_list:
                            N[num_view - 1][i] += 1
                        for valid_subject in cfg.HALL6_DATA.SUBJECTS_TEST:
                            print(valid_subject)
                            for act in vis_actions if EVAL else actions:
                                print(act)
                                absolute_path_gt = []
                                absolute_path_pred = []
                                poses_valid_2d = fetch([valid_subject], [act], is_test=True)
                                if len(poses_valid_2d[0]) == 0:
                                    print('No valid poses')
                                    continue
                                test_generator = ChunkedGenerator(cfg.TEST.BATCH_SIZE, poses_valid_2d, 1, pad=pad_t,
                                                                  causal_shift=causal_shift, shuffle=False,
                                                                  augment=False, kps_left=kps_left,
                                                                  kps_right=kps_right,
                                                                  joints_left=joints_left,
                                                                  joints_right=joints_right,
                                                                  extra_poses_3d=None)
                                for batch_2d, _, _ in test_generator.next_epoch():
                                    inputs = torch.from_numpy(batch_2d.astype('float32'))
                                    inputs_2d_gt = inputs[..., :2, :]
                                    inputs_2d_pre = inputs[..., 2:4, :]
                                    sub_action = [[valid_subject, act] for k in range(inputs.shape[0])]
                                    cam_3d = inputs[..., 4:7, :]
                                    vis = inputs[..., 7:8, :]
                                    inputs_3d_gt = cam_3d[:, pad_t:pad_t + 1]
                                    p3d_root = copy.deepcopy(inputs_3d_gt[:, :, :1])  # (B,T, 1, 3, N)
                                    absolute_path_gt.append(p3d_root)
                                    if not cfg.TRAIN.PREDICT_ROOT:
                                        inputs_3d_gt = inputs_3d_gt-p3d_root
                                    #inputs_3d_gt = inputs_3d_gt + p3d_root
                                    #p3d_gt_abs = inputs_3d_gt + p3d_root
                                    # inputs_3d_gt = inputs_3d_gt + p3d_root
                                    if use_2d_gt:
                                        h36_inp = inputs_2d_gt
                                        vis = torch.ones(*vis.shape)
                                    else:
                                        h36_inp = inputs_2d_pre

                                    inp = h36_inp[..., views_idx]  # B, T, V, C, N
                                    inp = torch.cat((inp, vis[..., views_idx]), dim=-2)
                                    B = inp.shape[0]
                                    if cfg.TEST.TRIANGULATE:
                                        trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t + 1, :, :2, :], valid_subject,
                                                                    views_idx)  # B, T, J, 3, N)
                                        loss = 0
                                        for idx, view_idx in enumerate(views_idx):
                                            loss_view_tmp = eval_metrc(cfg, trj_3d[..., idx],
                                                                       inputs_3d_gt[..., view_idx])
                                            loss += loss_view_tmp.item()
                                            action_mpjpe[act][num_view - 1][view_idx] += loss_view_tmp.item() *inputs_3d_gt.shape[0]

                                        action_mpjpe[act][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                                        continue

                                    inp_flip = copy.deepcopy(inp)
                                    inp_flip[:, :, :, 0] *= -1
                                    inp_flip[:, :, joints_left + joints_right] = inp_flip[:, :,
                                                                                 joints_right + joints_left]
                                    if cfg.NETWORK.USE_GT_TRANSFORM:
                                        inputs_3d_gt_flip = copy.deepcopy(inputs_3d_gt)
                                        inputs_3d_gt_flip[:, :, :, 0] *= -1
                                        inputs_3d_gt_flip[:, :, joints_left + joints_right] = inputs_3d_gt_flip[:, :,
                                                                                              joints_right + joints_left]
                                    if cfg.TEST.TEST_FLIP:
                                        if cfg.NETWORK.USE_GT_TRANSFORM:
                                            rotation = get_rotation(
                                                torch.cat((inputs_3d_gt, inputs_3d_gt_flip), dim=0)[..., views_idx])
                                            rotation = rotation.repeat(1, 1, 1, inp.shape[1], 1, 1)
                                        else:
                                            rotation = None
                                        confs, other_info = model_test(torch.cat((inp, inp_flip), dim=0), rotation)
                                        r_out = out

                                        out[B:, :, :, 0] *= -1
                                        out[B:, :, joints_left + joints_right] = out[B:, :, joints_right + joints_left]

                                        out = (out[:B] + out[B:]) / 2
                                    else:
                                        if cfg.NETWORK.USE_GT_TRANSFORM:
                                            rotation = get_rotation(inputs_3d_gt[..., views_idx])
                                            rotation = rotation.repeat(1, 1, 1, inp.shape[1], 1, 1)
                                        else:
                                            rotation = None
                                        confs, other_info = model_test(inp, rotation)
                                    confs = confs.permute(0, 1, 4, 2, 3).contiguous()

                                    out, stats_sing_values, _ = HumanCam.p2d_cam3d_batch_with_root(
                                        h36_inp[:, pad:pad + 1][...,views_idx], sub_action, views_idx, debug=False,
                                        confidences=confs[:, pad:pad + 1].permute(0, 1, 3, 4, 2).contiguous().to(h36_inp.device))

                                    absolute_path_pred.append(out[:, :, :1])
                                    prj_out_abs_to_2d = HumanCam.p3d_im2d_batch(out, sub_action, views_idx, with_distor=True, flip=batch_flip, gt_2d=inputs_2d_gt[:, pad:pad + 1,:, :].to(out.device))


                                    if id_eval == 0:
                                        np.save(args.visu_path+"/inputs_2d_gt" + "_epoch_" + str(epoch), inputs_2d_gt.detach().cpu().numpy())
                                        np.save(args.visu_path+"/prj_out_abs_to_2d" + "_epoch_" + str(epoch), prj_out_abs_to_2d.detach().cpu().numpy())
                                        np.save(args.visu_path+"/inputs_3d_gt" + "_epoch_" + str(epoch), inputs_3d_gt.detach().cpu().numpy())
                                        np.save(args.visu_path+"/out" + "_epoch_" + str(epoch), out.detach().cpu().numpy())

                                    out = out.detach().cpu()

                                    if EVAL and args.vis_3d:
                                        vis_tool.show(inputs_2d_pre[:, pad_t], out[:, 0], inputs_3d_gt[:, 0])

                                    if cfg.TEST.TEST_ROTATION:
                                        out = test_multi_view_aug(out, vis[..., views_idx])
                                        out[:, :, 0] = 0

                                    if cfg.NETWORK.USE_GT_TRANSFORM and EVAL and len(
                                            view_list) > 1 and cfg.TEST.ALIGN_TRJ:
                                        # TODO: 使用T帧姿态进行三角剖分得到平均骨骼长度再对齐
                                        trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t + 1, :, :2, :], valid_subject,
                                                                    views_idx)  # B, T, J, 3, N)
                                        out_align = align_target_numpy(cfg, out, trj_3d)
                                        out_align[:, :, 0] = 0
                                        out = out_align

                                    
                                    # if idx_eval == 0:
                                    #     for idx_, view_idx in enumerate(views_idx):
                                    #         fig = plt.figure("view_2d_" + str(idx_) + "_epoch_" + str(epoch))
                                    #         print("inputs_2d_gt.shape : " + str(inputs_2d_gt.shape))
                                    #         h_inp_2d = inp[:, pad:pad + 1, :, :][0, 0, :, 0,idx_].detach().numpy()
                                    #         w_inp_2d = inp[:, pad:pad + 1, :, :][0, 0, :, 1,idx_].detach().numpy()
                                    #         plt.scatter(h_inp_2d, w_inp_2d, color='r', marker='o', label="input_2d")
                                    #         fig = plt.figure("view " + str(idx_))
                                    #         ax = fig.add_subplot(projection='3d')
                                    #
                                    #         print("p3d_gt_abs.shape : "+str(p3d_gt_abs.shape))
                                    #         x_out = out[0, 0, :, 0, idx_].detach().cpu().numpy()
                                    #         y_out = out[0, 0, :, 1, idx_].detach().cpu().numpy()
                                    #         z_out = out[0, 0, :, 2, idx_].detach().cpu().numpy()
                                    #
                                    #         x_gt = inputs_3d_gt[0, 0, :, 0, view_idx].detach().cpu().numpy()
                                    #         y_gt = inputs_3d_gt[0, 0, :, 1, view_idx].detach().cpu().numpy()
                                    #         z_gt = inputs_3d_gt[0, 0, :, 2, view_idx].detach().cpu().numpy()
                                    #         ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
                                    #         ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
                                    #         ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
                                    #         plt.savefig("view " + str(v))
                                    #     plt.show()

                                    #build data npz
                                    prj_out_abs_to_2d = HumanCam.p3d_im2d_batch(out, sub_action, view_list,
                                                                                with_distor=True, flip=batch_flip,
                                                                                gt_2d=inputs_2d_gt[:, pad:pad + 1, :,
                                                                                      :].to(out.device))

                                    pose_2D_from3D_gt = prj_out_abs_to_2d.permute(0, 2, 3, 4, 1).contiguous()
                                    for i, s_a in enumerate(sub_action):
                                        subject = s_a[0]
                                        action = s_a[1]
                                        if subject not in data_npy.keys():
                                            data_npy[subject] = {}
                                        if action not in data_npy[subject].keys():
                                            data_npy[subject][action] = []
                                            for v in range(len(view_list)):
                                                data_npy[subject][action].append([])
                                        for v in range(len(view_list)):
                                            # print(pose_2D_from3D_gt.shape)
                                            curr_data = torch.cat([pose_2D_from3D_gt[i, 0, :, :, v].cpu(),
                                                                   inp[:, :, :, :2, :][i, pad, :, :, v].cpu(),
                                                                   out[i, 0, :, :, v].cpu(),
                                                                   inp[:, :, :, -1:, :][i, pad, :, :, v].cpu()], dim=-1)
                                            data_npy[subject][action][v].append(curr_data)

                                    #detect nan in out and create a mask
                                    mask = torch.isnan(out).any(dim=-1).any(dim=-1).any(dim=-1).any(dim=-1)
                                    # remove nan from out
                                    out = out[~mask]
                                    # remove nan from inputs_3d_gt
                                    inputs_3d_gt = inputs_3d_gt[~mask]
                                    sub_action = np.array(sub_action)[~mask].tolist()

                                    #end build data npz
                                    idx_eval +=1
                                    loss = 0
                                    bone_loss = 0
                                    for idx_, view_idx in enumerate(views_idx):
                                        loss_view_tmp = eval_metrc(cfg, out[..., idx_], inputs_3d_gt[..., view_idx])
                                        #bone_pred_len = bone_losses(out[..., idx_:idx_+1].permute((0,1,4,2,3)).contiguous(), bones.cpu(), cfg.HALL6_DATA.SUBJECTS_TEST, batch_subjects=sub_action, cfg=cfg)[0]
                                        #bone_error = torch.mean(torch.abs(torch.squeeze(bone_pred_len)-bones_means.cpu()))
                                        bone_error = bone_len_loss(gt_bones_lens,out.permute(0, 1, 4, 2, 3).contiguous(), bones.to(out.device),cfg.HALL6_DATA.SUBJECTS_TRAIN,batch_subjects=sub_action,cfg=cfg,std_bones_len_prior=bones_stds.to(out.device))
                                        loss += loss_view_tmp.item()
                                        bone_loss += bone_error.item()
                                        action_mpjpe[act][num_view - 1][idx_] += loss_view_tmp.item() * inputs_3d_gt.shape[0]
                                        action_bone_error[act][num_view - 1][idx_] += bone_error.item() * inputs_3d_gt.shape[0]
                                    action_mpjpe[act][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                                    action_bone_error[act][num_view - 1][-1] += bone_loss * inputs_3d_gt.shape[0]
                                if cfg.TRAIN.PREDICT_ROOT:
                                    if id_traj == 0:
                                        absolute_path_gt = torch.cat(absolute_path_gt, dim=0).detach().cpu().numpy()
                                        absolute_path_pred = torch.cat(absolute_path_pred, dim=0).detach().cpu().numpy()
                                        np.save(args.visu_path+"/absolute_path_pred" + "_epoch_" + str(epoch), absolute_path_pred)
                                        np.save(args.visu_path+"/absolute_path_gt" + "_epoch_" + str(epoch), absolute_path_gt)
                                    id_traj += 1
                print(view_list)
                n_sub = 0
                n_act = []
                for subject in data_npy.keys():
                    n_sub += 1
                    n_act.append(0)
                    for action in data_npy[subject].keys():
                        n_act[-1] += 1
                        for v in range(len(view_list)):
                            data_npy[subject][action][v] = torch.stack(data_npy[subject][action][v], dim=0).numpy()
                print("n_sub", n_sub)
                print("n_act", n_act)

                np.savez(args.visu_path+  '/' + 'data_epoch_' + str(epoch)+".npz", **data_npy)
                                
                for num_view in cfg.TEST.NUM_VIEWS:
                    tmp = [0] * (NUM_VIEW + 1)
                    print('num_view:{}'.format(num_view))
                    for act in action_bone_error:
                        if action_frames[act] > 0:
                            for i in range(NUM_VIEW):
                                action_bone_error[act][num_view - 1][i] /= (action_frames[act] * N[num_view - 1][i])
                            action_bone_error[act][num_view - 1][-1] /= (action_frames[act] * N[num_view - 1][-1] * num_view)
                            print('bone err of {:18}'.format(act), end=' ')
                            for i in range(NUM_VIEW):
                                print('view_{}: {:.3f}'.format(cfg.HALL6_DATA.TEST_CAMERAS[i],
                                                               action_bone_error[act][num_view - 1][i] * 1000), end='    ')
                                tmp[i] += action_bone_error[act][num_view - 1][i] * 1000
                            print('avg_action: {:.3f}'.format(action_bone_error[act][num_view - 1][-1] * 1000))
                            tmp[-1] += action_bone_error[act][num_view - 1][-1] * 1000
                    print('avg:', end='                        ')
                    for i in range(NUM_VIEW):
                        print('view_{}: {:.3f}'.format(i, tmp[i] / len(action_frames)), end='    ')
                        summary_writer.add_scalar(
                            "test_bone_err_t{}_n_view_{}_view_{}/epoch".format(t_len, num_view, i),
                            tmp[i] / len(action_frames), epoch)
                    print('avg_all   : {:.3f}'.format(tmp[-1] / len(action_frames)))

                    if summary_writer is not None:
                        summary_writer.add_scalar("test_bone_err_t{}_v{}/epoch".format(t_len, num_view),
                                                  tmp[-1] / len(action_frames), epoch)
                    epoch_bone_valid += tmp[-1] / len(action_frames)
                epoch_bone_valid /= len(cfg.TEST.NUM_VIEWS)
                print('t_len:{} avg:{:.3f}'.format(t_len, epoch_bone_valid))
                if summary_writer is not None:
                    summary_writer.add_scalar("epoch_bone_valid/epoch", epoch_bone_valid, epoch)
                    
                    
                print('num_actions :{}'.format(len(action_frames)))
                for num_view in cfg.TEST.NUM_VIEWS:
                    tmp = [0] * (NUM_VIEW + 1)
                    print('num_view:{}'.format(num_view))
                    for act in action_mpjpe:
                        if action_frames[act] > 0:
                            for i in range(NUM_VIEW):
                                action_mpjpe[act][num_view - 1][i] /= (action_frames[act] * N[num_view - 1][i])
                            action_mpjpe[act][num_view - 1][-1] /= (action_frames[act] * N[num_view - 1][-1] * num_view)
                            print('mpjpe of {:18}'.format(act), end=' ')
                            for i in range(NUM_VIEW):
                                print('view_{}: {:.3f}'.format(cfg.HALL6_DATA.TEST_CAMERAS[i],
                                                               action_mpjpe[act][num_view - 1][i] * 1000), end='    ')
                                tmp[i] += action_mpjpe[act][num_view - 1][i] * 1000
                            print('avg_action: {:.3f}'.format(action_mpjpe[act][num_view - 1][-1] * 1000))
                            tmp[-1] += action_mpjpe[act][num_view - 1][-1] * 1000
                    print('avg:', end='                        ')
                    for i in range(NUM_VIEW):
                        print('view_{}: {:.3f}'.format(i, tmp[i] / len(action_frames)), end='    ')
                        summary_writer.add_scalar(
                            "test_mpjpe_t{}_n_view_{}_view_{}/epoch".format(t_len, num_view, i),
                            tmp[i] / len(action_frames), epoch)
                    print('avg_all   : {:.3f}'.format(tmp[-1] / len(action_frames)))

                    if summary_writer is not None:
                        summary_writer.add_scalar("test_mpjpe_t{}_v{}/epoch".format(t_len, num_view),
                                                  tmp[-1] / len(action_frames), epoch)
                    epoch_loss_valid += tmp[-1] / len(action_frames)
                epoch_loss_valid /= len(cfg.TEST.NUM_VIEWS)
                print('t_len:{} avg:{:.3f}'.format(t_len, epoch_loss_valid))
                if summary_writer is not None:
                    summary_writer.add_scalar("epoch_loss_valid/epoch", epoch_loss_valid, epoch)
                    
            

            if EVAL:
                eval_elapsed = (time.time() - eval_start_time) / 60
                print('time:{:.2f}'.format(eval_elapsed))
                exit()

        epoch += 1

        if epoch_loss_valid < best_result:
            best_result = epoch_loss_valid
            best_state_dict = copy.deepcopy(model.module.state_dict())
            best_result_epoch = epoch
        elapsed = (time.time() - start_time) / 60
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

