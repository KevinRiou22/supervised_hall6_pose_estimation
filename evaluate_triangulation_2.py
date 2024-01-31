import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
from common.loss import *
import torch
from common.config import config as cfg
from common.arguments import parse_args
from common.config import reset_config, update_config
from collections import OrderedDict
import matplotlib
import seaborn as sns

font = {'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

args = parse_args()
update_config(args.cfg)  ###config file->cfg
reset_config(cfg, args)

path_dataset = "./data/"
#path_dataset ="visu/submit/learn_conf_pred/"
path_meta_data = "./data/"
#operators = [1]

operators = [1, 2, 3, 4, 5]
tasks= [0, 1, 2, 3]
examples = list(range(1, 31))

my_data_pth = path_dataset+args.data_name+".npz"#'hall6.npz' ####one example:t1_o1_ex7
data_npy = np.load(my_data_pth, allow_pickle=True)
data_npy = dict(data_npy)



f = open('data/cams_params_all_examples.json', )
params = json.load(f)

cams = params['cams_order'] #os.listdir('data/images/task{}/operator{}/example{}'.format(k_a[0], operator_str, k_a[1]))
frame = 0

f = open('data/cams_params_all_examples.json', )

params = json.load(f)
# params = dict(params)
#print(params)
cameras_intrinsic_params = params['h36m_cameras_intrinsic_params']
resolutions={}
for cam_dict in cameras_intrinsic_params:
    resolutions[cam_dict['id']] = [cam_dict['res_w'], cam_dict['res_h']]

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

print("bones : " + str(bones))
#bones_stds = torch.from_numpy(np.std(np.array(bones_means_), axis=(0)))#.cuda()
bones_means = torch.from_numpy(np.load("data/bones_length_hall6_triang_measure.npy")) # .cuda()
#bones_stds = torch.std(torch.from_numpy(np.load("data/bones_length_hall6_triang_measure.npy")), axis=0)
print("bones_means : " + str(bones_means))
#enumerate data_npy, keys and values
reprojection_errors_subjects = [[] for i in range(5)]
reprojection_errors_examples = []
reprojection_errors_views = [[] for i in range(16)]

bone_3D_std_examples = []
reprojection_errors_examples = []
bones_3D_errors_examples = []

#initialize array with 'nan' values
overall_bones_3D_errors = np.empty((len(operators), len(tasks), len(examples), len(cfg.HALL6_DATA.TEST_CAMERAS), 10000, len(bone_names_in_bones_list)))
overall_bones_3D_errors[:] = np.nan
#print(overall_bones_3D_errors)
indexes = []


for i, (k_s, v_s) in enumerate(data_npy.items()):
    print(k_s)
    operator_id = int((''.join(filter(str.isdigit, k_s)))) - 1
    #operator_str = "operator" + operator_id
    bone_3D_std_examples.append([])
    bones_3D_errors_examples.append([])
    reprojection_errors_examples.append([])
    for j, (k_a, v_a) in enumerate(v_s.item().items()):
        k_a = k_a.split("_")

        task_id = int((''.join(filter(str.isdigit, k_a[0]))))-1
        example_id = int((''.join(filter(str.isdigit, k_a[1]))))-1

        cams_videos = []
        # if k_a[1]!="example1":
        #     continue
        print(k_a)
        # remove x and y ticks and labels
        reprojection_errors_examples[-1].append([])
        bone_3D_std_examples[-1].append([])
        bones_3D_errors_examples[-1].append([])
        for idx, view_idx in enumerate(cfg.HALL6_DATA.TEST_CAMERAS):
            cam_resolution = resolutions[cfg.HALL6_DATA.CAMERAS_IDS[view_idx]]
            poses_2D_hrnet = (torch.from_numpy(v_a[idx][:,:, 2:4])+1)/2#frame, joints, (x,y)
            poses_2D_hrnet[:,:, 0] = poses_2D_hrnet[:,:, 0]*cam_resolution[0]
            poses_2D_hrnet[:,:, 1] = poses_2D_hrnet[:,:, 1]*cam_resolution[1]
            poses_2D_reprojection =  (torch.from_numpy(v_a[idx][:,:, 0:2])+1)/2 #frame, joints, (x,y)
            poses_2D_reprojection[:,:, 0] = poses_2D_reprojection[:,:, 0]*cam_resolution[0]
            poses_2D_reprojection[:,:, 1] = poses_2D_reprojection[:,:, 1]*cam_resolution[1]
            poses_3D = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(v_a[idx][: ,:, 4:7]), 0), -1) #frame, joints, (x,y,z)
            confidences = v_a[idx][: ,:, 7] #frame, joints, 1
            sub_action = [[k_s, k_a]]
            #pred_bone_mean, pred_bone_std = bone_losses(poses_3D.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones.to(poses_3D.device), cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg)
            bones_err = bone_len_mae(gt_bones_lens, poses_3D.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones.to(poses_3D.device), cfg.HALL6_DATA.SUBJECTS_TRAIN,batch_subjects=sub_action, avg_ov_frames=False)
            #print("bones_err.shape : " + str(bones_err.shape))
            #print("torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list))" + str(torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list)).shape))
            #overall_bones_3D_errors = np.append(overall_bones_3D_errors, torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list)), axis=0)
            overall_bones_3D_errors[operator_id, task_id, example_id, idx, :bones_err.shape[1], :] = torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list))

#np.save("overall_bones_3D_errors.npy", overall_bones_3D_errors)
#get the mean of 3D errors for each subject, for each bone, avoiding nan values
mean_overall_bones_3D_errors = np.nanmean(overall_bones_3D_errors, axis=(1, 2, 3, 4))
print("mean_overall_bones_3D_errors : " + str(mean_overall_bones_3D_errors))
#get the worst frame in overall_bones_3D_errors
worst_frame = np.unravel_index(np.nanargmax(overall_bones_3D_errors, axis=None), overall_bones_3D_errors.shape)
# get the index of the max value of the 6D array

print("worst_frame : " + str(worst_frame))
print("worst frame error : ",overall_bones_3D_errors[worst_frame])
#merge axis 0, 1, 2, 3, 4 in overall_bones_3D_errors
per_bone_errors = np.reshape(overall_bones_3D_errors, (-1, len(bone_names_in_bones_list)))
#remove nan values
per_bone_errors = per_bone_errors[~np.isnan(per_bone_errors).any(axis=1)]
print("per_bone_errors : " + str(per_bone_errors.shape))
print("per_bone_errors.shape : " + str(per_bone_errors.shape))
# violin plot of per bone errors, avoiding nan values
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Violin plot of per bone errors')
ax.set_ylabel('Error (mm)')
ax.set_xlabel('Bone')
ax.set_xticks(np.arange(len(bone_names_in_bones_list)))
ax.set_xticklabels(bone_names_in_bones_list)
#plot violin with plt
plt.violinplot(per_bone_errors, showmeans=True)

plt.show()
