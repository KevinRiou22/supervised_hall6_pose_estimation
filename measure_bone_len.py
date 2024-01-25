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

font = {'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

args = parse_args()
update_config(args.cfg)  ###config file->cfg
reset_config(cfg, args)

path_dataset = "./data/"
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
gt_bones_lens = json.load(f)
bone_names_h36m = cfg.HALL6_DATA.BONES_NAMES
bones_h36m = cfg.HALL6_DATA.BONES
bones_means_ = []
bones = []
count_subj=0
symmetry_bones = [[],[]]
bone_names_in_bones_list = []
for sub_id in gt_bones_lens.keys():
    sub_processed = gt_bones_lens[sub_id]['h36m']
    print(sub_processed)
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
    count_subj +=1
cfg.HALL6_DATA.BONES_SYMMETRY = symmetry_bones
print("bone_names_in_bones_list : " + str(bone_names_in_bones_list))
print("cfg.HALL6_DATA.BONES_SYMMETRY : " + str(cfg.HALL6_DATA.BONES_SYMMETRY))
torch.from_numpy(np.array(bones_means_))#.cuda()
bones_means = torch.from_numpy(np.mean(np.array(bones_means_), axis=(0)))#.cuda()
bones = torch.from_numpy(np.array(bones))#.cuda()
print("bones_means : " + str(bones_means))
print("bones : " + str(bones))
bones_stds = torch.from_numpy(np.std(np.array(bones_means_), axis=(0)))#.cuda()


#enumerate data_npy, keys and values
reprojection_errors_subjects = [[] for i in range(5)]
reprojection_errors_examples = []
reprojection_errors_views = [[] for i in range(16)]

bone_3D_std_examples = []
reprojection_errors_examples = []
bones_3D_errors_examples = []

triangulated_bones = []
error_to_measure = []
for i, (k_s, v_s) in enumerate(data_npy.items()):
    print(k_s)
    bone_3D_std_examples.append([])
    bones_3D_errors_examples.append([])
    reprojection_errors_examples.append([])
    sub_error_to_measure = []
    sub_triangulated_bones = []
    for j, (k_a, v_a) in enumerate(v_s.item().items()):
        k_a = k_a.split("_")
        if k_a[0]!="task1" or k_a[1]!="example1":
            continue
        operator_str = "operator" + (''.join(filter(str.isdigit, k_s)))
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
            poses_3D = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(v_a[idx][:1 ,:, 4:7]), 0), -1) #frame, joints, (x,y,z)
            sub_action = [[k_s, k_a]]
            pred_bone_mean, pred_bone_std = bone_losses(poses_3D.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones, cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg)
            # print("pred_bone_mean : " + str(pred_bone_mean))
            # print("bones_means : " + str(bones_means))
            # input()
            sub_triangulated_bones.append(torch.squeeze(pred_bone_mean).detach().cpu().numpy())
            bones_err = torch.abs(torch.squeeze(pred_bone_mean)-bones_means).detach().cpu().numpy()
            sub_error_to_measure.append(bones_err)
    triangulated_bones.append(np.mean(np.array(sub_triangulated_bones), axis=0))
    error_to_measure.append(np.mean(np.array(sub_error_to_measure), axis=0))


triangulated_bones = np.array(triangulated_bones)
error_to_measure = np.array(error_to_measure)

print("triangulated_bones : " + str(triangulated_bones))
print("error_to_measure : " + str(error_to_measure))
print(triangulated_bones.shape)
print(np.array(bones_means_).shape)
#print(bones_stds.shape)
np.save("data/bones_length_hall6_triang_measure.npy", triangulated_bones)
np.save("data/bones_length_hall6_physical_measure.npy", np.array(bones_means_))

