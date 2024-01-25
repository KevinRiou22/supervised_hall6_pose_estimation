import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
import json
from common.loss import *
import torch
from common.config import config as cfg
from common.arguments import parse_args
from common.config import reset_config, update_config
from collections import OrderedDict
import matplotlib

args = parse_args()
update_config(args.cfg)  ###config file->cfg
reset_config(cfg, args)

#path_dataset = "./data/"
path_dataset ="visu/submit/learn_conf_pred/"
path_meta_data = "./data/"
#operators = [1]

operators = [5]
tasks= [1]
examples = [2]#list(range(1, 31))

#my_data_pth = path_dataset+"triangulated_3D_16_cams.npz"#"triangulated_3D_16_cams.npz"#'hall6.npz' ####one example:t1_o1_ex7
my_data_pth = path_dataset+args.data_name+".npz"
data_npy = np.load(my_data_pth, allow_pickle=True)
data_npy = dict(data_npy)



def read_video_cv2(video_path, frame_range=[0, 10]):
    cap = cv2.VideoCapture(video_path)
    all = []
    i = 0
    while cap.isOpened() and i < frame_range[1]:
        ret, frame = cap.read()
        if ret == True:
            # rgb to bgr for opencv
            if i>=frame_range[0] and i<frame_range[1]:
                frame = frame[:, :, ::-1]

                arr = np.array(frame)
                all.append(arr)
            i += 1
        else:
             break
    return np.array(all)
f = open('data/cams_params_all_examples.json', )
params = json.load(f)

cams = params['cams_order'] #os.listdir('data/images/task{}/operator{}/example{}'.format(k_a[0], operator_str, k_a[1]))
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
#bones_means = torch.from_numpy(np.mean(np.array(bones_means_), axis=(0)))#.cuda()
bones = torch.from_numpy(np.array(bones))#.cuda()
#print("bones_means : " + str(bones_means))
#print("bones : " + str(bones))
#bones_stds = torch.from_numpy(np.std(np.array(bones_means_), axis=(0)))#.cuda()
bones_means = torch.mean(torch.from_numpy(np.load("data/bones_length_hall6_triang_measure.npy")), axis=0)  # .cuda()
bones_stds = torch.std(torch.from_numpy(np.load("data/bones_length_hall6_triang_measure.npy")), axis=0)




frame = 0

op5_task1_ex2 = []
op1_task1_ex1 = []

#enumerate data_npy, keys and values
for i, (k_s, v_s) in enumerate(data_npy.items()):
    print(k_s)
    for j, (k_a, v_a) in enumerate(v_s.item().items()):
        k_a = k_a.split("_")
        operator_str = "operator" + (''.join(filter(str.isdigit, k_s)))

        cams_videos = []

        if k_a[1]!="example1" or k_a[0]!="task1" or operator_str!="operator1":
            continue

        #for cam in cams:
        for idx, view_idx in enumerate(cfg.HALL6_DATA.TEST_CAMERAS):
            cam = cams[view_idx]
            cams_videos.append(read_video_cv2('data/images/{}/{}/{}/{}/video.avi'.format(k_a[0], operator_str, k_a[1], cam), [frame,frame+1]))
        print(k_a)
        fig1, axs = plt.subplots(4, 4)
        # remove x and y ticks and labels
        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])

        #fig2, axs_3D = plt.subplots(4, 4)
        # # remove x and y ticks and labels
        # for ax in axs_3D.flat:
        #     ax.set(xticks=[], yticks=[])

        i = 0
        figs_3D = []
        for idx, view_idx in enumerate(cfg.HALL6_DATA.TEST_CAMERAS):
            print(v_a[idx].shape)
            #print(v_a[v][0,:, :2])
            #print(v_a[v][0, :, 2:4])
            axs[i // 4, i % 4].set_title(cams[view_idx]+" / "+str(idx))
            first_frame = cams_videos[i][0]
            # plot first frame in subplot i
            axs[i // 4, i % 4].imshow(first_frame)


            h_inp_2d = v_a[idx][frame,:, 2]
            w_inp_2d = v_a[idx][frame, :, 3]


            h_3D_to_2D = v_a[idx][frame, :, 0]
            w_3D_to_2D = v_a[idx][frame, :, 1]

            if "r" in cams[view_idx] or "l" in cams[view_idx]:
                print("zed")
                h_inp_2d = h_inp_2d * 1920/2 + 1920/2
                w_inp_2d = w_inp_2d * 1080/2 + 1080/2
                h_3D_to_2D = h_3D_to_2D * 1920/2 + 1920/2
                w_3D_to_2D = w_3D_to_2D * 1080/2 + 1080/2
            else:
                print("flir")
                h_inp_2d = h_inp_2d * 2048/2 + 2048/2
                w_inp_2d = w_inp_2d * 2048/2 + 2048/2
                h_3D_to_2D = h_3D_to_2D * 2048/2 + 2048/2
                w_3D_to_2D = w_3D_to_2D * 2048/2 + 2048/2

            axs[i // 4, i % 4].scatter(h_inp_2d, w_inp_2d, color='r', marker='.', label="input_2d")
            axs[i // 4, i % 4].scatter(h_3D_to_2D, w_3D_to_2D, color='g', marker='+', label="prj_3dgt_abs_to_2d")

            x_out = v_a[idx][frame, :, 4]
            y_out = v_a[idx][frame, :, 5]
            z_out = v_a[idx][frame, :, 6]
            figs_3D.append(plt.figure("3D view {}".format(idx)))
            axs_3D = figs_3D[-1].add_subplot(projection='3d')
            axs_3D.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
            #axs_3D.set_box_aspect((np.ptp(x_out), np.ptp(y_out), np.ptp(z_out)))
            #set all axis limits to [-4, 4]
            axs_3D.set_xlim3d([-1, 1])
            axs_3D.set_ylim3d([-1, 1])
            axs_3D.set_zlim3d([3, 5])

            i += 1
        plt.legend()
        plt.show()