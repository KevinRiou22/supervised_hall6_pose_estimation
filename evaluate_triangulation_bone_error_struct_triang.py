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



f = open('data/cams_params_all_examples_.json', )
params = json.load(f)

cams = params['cams_order'] #os.listdir('data/images/task{}/operator{}/example{}'.format(k_a[0], operator_str, k_a[1]))
frame = 0

f = open('data/cams_params_all_examples_.json', )

params = json.load(f)
# params = dict(params)
#print(params)
cameras_intrinsic_params = params['h36m_cameras_intrinsic_params']
resolutions={}
for cam_dict in cameras_intrinsic_params:
    resolutions[cam_dict['id']] = [cam_dict['res_w'], cam_dict['res_h']]

gt_bones_lens={}
processed_lens = np.load("data/bones_length_hall6_triang_measure_16_cams_h36m_struct.npy")

h36m_hall6_common_bones_names = ['rhip', 'rknee', 'rfoot', 'lhip', 'lknee', 'lfoot', 'lelbow', 'lwrist', 'relbow', 'rwrist']#
#get indexes fo h36m_hall6_common_bones in cfg.H36M_DATA.BONES_NAMES
bones_names = h36m_hall6_common_bones_names#cfg.H36M_DATA.BONES_NAMES
h36m_hall6_common_bones = [bones_names.index(x) for x in h36m_hall6_common_bones_names]
for sub_id in range(processed_lens.shape[0]):
    gt_bones_lens["S"+str(sub_id+1)] = torch.from_numpy(processed_lens[sub_id])[h36m_hall6_common_bones]

bones = torch.from_numpy(np.array(cfg.H36M_DATA.BONES))[h36m_hall6_common_bones]
print(bones)

#enumerate data_npy, keys and values
reprojection_errors_subjects = [[] for i in range(5)]
reprojection_errors_examples = []
reprojection_errors_views = [[] for i in range(16)]

bone_3D_std_examples = []
reprojection_errors_examples = []
bones_3D_errors_examples = []

#initialize array with 'nan' values
overall_bones_3D_errors = np.empty((len(operators), len(tasks), len(examples), len(cfg.HALL6_DATA.TEST_CAMERAS), 8000, len(bones_names)))
overall_bones_3D_errors[:] = np.nan
#print(overall_bones_3D_errors)
indexes = []

failed_triangulations = 0
total_triangulations = 0

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
            poses_2D_hrnet[:, :, 0] = poses_2D_hrnet[:,:, 0]*cam_resolution[0]
            poses_2D_hrnet[:, :, 1] = poses_2D_hrnet[:,:, 1]*cam_resolution[1]
            poses_2D_reprojection =  (torch.from_numpy(v_a[idx][:,:, 0:2])+1)/2 #frame, joints, (x,y)
            poses_2D_reprojection[:, :, 0] = poses_2D_reprojection[:, :, 0]*cam_resolution[0]
            poses_2D_reprojection[:, :, 1] = poses_2D_reprojection[:, :, 1]*cam_resolution[1]
            poses_3D = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(v_a[idx][: ,:, 4:7]), 0), -1) #frame, joints, (x,y,z)
            confidences = v_a[idx][: ,:, 7] #frame, joints, 1
            sub_action = [[k_s, k_a]]
            #pred_bone_mean, pred_bone_std = bone_losses(poses_3D.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones.to(poses_3D.device), cfg.HALL6_DATA.SUBJECTS_TRAIN, batch_subjects=sub_action, cfg=cfg)
            bones_err = bone_len_mae(gt_bones_lens, poses_3D.permute(0, 1, 4, 2, 3).contiguous()[:, :, :], bones.to(poses_3D.device), cfg.HALL6_DATA.SUBJECTS_TRAIN,batch_subjects=sub_action, avg_ov_frames=False, remove_fails_from_stats=args.remove_fails_from_stats)
            #print("torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list))" + str(torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list)).shape))
            #overall_bones_3D_errors = np.append(overall_bones_3D_errors, torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list)), axis=0)
            # print(bones_err.shape)
            # input()
            failed_triangulations += torch.sum(torch.isnan(bones_err))
            print(failed_triangulations)
            total_triangulations += bones_err.shape[0]*bones_err.shape[1]*bones_err.shape[2]*bones_err.shape[3]
            overall_bones_3D_errors[operator_id, task_id, example_id, idx, :bones_err.shape[1], :] = torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bones_names))*1000

failed_triangulations = 100*failed_triangulations/total_triangulations
print("failed_triangulations : " + str(failed_triangulations))
#get the mean of 3D errors for each subject, avoiding nan values
mean_overall_bones_3D_errors = np.nanmean(overall_bones_3D_errors, axis=(1, 2, 3, 4, 5))
print("mean bone error for each subject : " + str(mean_overall_bones_3D_errors))
#get the confidence intervals on the 3D errors for each subject, avoiding nan values
#conf_int_overall_bones_3D_errors = np.nanpercentile(overall_bones_3D_errors, [2.5, 97.5], axis=(1, 2, 3, 4, 5))
#print("conf_int_overall_bones_3D_errors : " + str(conf_int_overall_bones_3D_errors))
#get the worst frame in overall_bones_3D_errors
worst_frame = np.unravel_index(np.nanargmax(overall_bones_3D_errors, axis=None), overall_bones_3D_errors.shape)
# get the index of the max value of the 6D array

print("worst_frame : " + str(worst_frame))
print("worst frame error : ",overall_bones_3D_errors[worst_frame])
#merge axis 0, 1, 2, 3, 4 in overall_bones_3D_errors
per_bone_errors = np.reshape(overall_bones_3D_errors, (-1, len(bones_names)))
#remove nan values
per_bone_errors = per_bone_errors[~np.isnan(per_bone_errors).any(axis=1)]
print("per_bone_errors : " + str(per_bone_errors.shape))
print("per_bone_errors.shape : " + str(per_bone_errors.shape))

mean_overall_bones_3D_errors = ["{:.1f}".format(i) for i in mean_overall_bones_3D_errors.tolist()]
#display scientific notation with 2 digits if more than 2 digits before the exponent, otherwise display 2 digits
mean_overall_bones_3D_errors = [str(i) if len(i.split(".")[0])<=2 else "{:.2e}".format(float(i)) for i in mean_overall_bones_3D_errors]

worst_error = ["{:.1f}".format(overall_bones_3D_errors[worst_frame]) if len("{:.1f}".format(overall_bones_3D_errors[worst_frame]).split(".")[0])<=2 else "{:.2e}".format(float("{:.1f}".format(overall_bones_3D_errors[worst_frame])))]
failed_triangulations = ["{:.1f}".format(failed_triangulations.detach().cpu().numpy())]
global_mean = ["{:.1f}".format(np.nanmean(per_bone_errors))]
global_mean = [str(i) if len(i.split(".")[0])<=2 else "{:.2e}".format(float(i)) for i in global_mean]
to_print = ["16 views"] + mean_overall_bones_3D_errors + global_mean + worst_error+[" "]+["-" for i in range(7)]+[" "]+failed_triangulations
#convert all elements in to_print to string
to_print = [str(elem) for elem in to_print]
print(" & ".join(to_print))


# violin plot of per bone errors, avoiding nan values
fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_title('Violin plot of bone errors,\n while triangulation from 16 views,\n weighted by 2D detector confidences.')
#ax.set_ylabel('Error (mm)')
#ax.set_xlabel('Bones')
#set xlabel, ylabel and title fonts to bold
ax.set_xlabel('Bones', fontweight='bold')
ax.set_ylabel('Error (mm)', fontweight='bold')
ax.set_title('Violin plot of bone errors,\n while triangulating from 16 views,\n weighted by 2D detector confidences.', fontweight='bold')
#replace "Right" by R. and "Left" by L. in bone_names_in_bones_list
#bone_names_in_bones_list = [bone_name.replace("Right", "R.").replace("Left", "L.") for bone_name in cfg.H36M_DATA.BONES_NAMES]
# replace "l_" by L. and "r_" by R. in bone_names_in_bones_list
#bone_names_in_bones_list = [bone_name.replace("l_e", "L.e").replace("r_e", "R.e") for bone_name in cfg.H36M_DATA.BONES_NAMES]
#replace "_" by "\n" in bone_names_in_bones_list
#bone_names_in_bones_list = [bone_name.replace("_", "\n") for bone_name in bone_names_in_bones_list]

#plot the boxplot with plt
plt.boxplot(per_bone_errors, showmeans=False, showfliers=False)


#plot violin with plt
plt.violinplot(per_bone_errors, showmeans=True, showextrema=False)
#set xticks
ax.set_xticks(np.arange(len(bones_names))+1)
ax.set_xticklabels(bones_names)
#reduce font size of xticks
plt.xticks(fontsize=8)
#write xtick obliquely
plt.xticks(rotation=45)
# limit y axis to 0-300
plt.ylim(0, 300)
# add y grid
plt.grid(axis='y')
#add subticks y axis
ax.tick_params(axis='y', which='minor', bottom=False)
#add a vertical line for each sub-tick
for i in range(0, 301, 10):
    plt.axhline(y=i, color='grey', linestyle='-', linewidth=0.5)


plt.show()
