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

#initialize empty numpy array
overall_bones_3D_errors = np.empty((0, len(bone_names_in_bones_list)))
indexes = []


for i, (k_s, v_s) in enumerate(data_npy.items()):
    print(k_s)
    bone_3D_std_examples.append([])
    bones_3D_errors_examples.append([])
    reprojection_errors_examples.append([])
    for j, (k_a, v_a) in enumerate(v_s.item().items()):
        k_a = k_a.split("_")
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
            overall_bones_3D_errors = np.append(overall_bones_3D_errors, torch.squeeze(bones_err).detach().cpu().numpy().reshape(-1, len(bone_names_in_bones_list)), axis=0)

            curent_indexes = [[k_s, k_a, view_idx, frame_id] for frame_id in range(poses_3D.shape[1])]
            indexes = indexes + curent_indexes
            bones_err=torch.mean(bones_err, dim=1)
            # print("pred_bone_mean : " + str(pred_bone_mean))
            # print("bones_means : " + str(bones_means))
            # input()

            #bones_err = torch.abs(torch.squeeze(pred_bone_mean)-bones_means[i]).detach().cpu().numpy()
            bones_3D_errors_examples[-1][-1].append(torch.squeeze(bones_err).detach().cpu().numpy())
            reproj_error = mpjpe(poses_2D_hrnet, poses_2D_reprojection)
            reproj_error = reproj_error.detach().cpu().numpy()
            reprojection_errors_views[idx].append(reproj_error)
            reprojection_errors_subjects[i].append(reproj_error)
            reprojection_errors_examples[-1][-1].append(reproj_error)
            # print(pred_bone_std.shape)
            # print(pred_bone_mean)
            # input()
            #bone_3D_std_examples[-1][-1].append(np.squeeze(pred_bone_std.detach().cpu().numpy()))

print("overall max error : " + str(np.max(overall_bones_3D_errors, axis=0)))
print("overall min error : " + str(np.min(overall_bones_3D_errors, axis=0)))
print(overall_bones_3D_errors.shape)
print(len(indexes))
print(np.argmax(overall_bones_3D_errors))
print(indexes[np.argmax(overall_bones_3D_errors, axis=0)])

# violin plot of overall_bones_3D_errors
plt.figure("overall_bones_3D_errors")
plt.violinplot(overall_bones_3D_errors, showmeans=True, showextrema=False)
plt.xticks(np.arange(1, len(bone_names_in_bones_list)+1), bone_names_in_bones_list)
plt.ylabel("mm")
plt.title("overall_bones_3D_errors")
plt.grid()
plt.show()


avg_repr_error = []
to_print_bone = []
to_print_reproj = []
for i in range(len(bones_3D_errors_examples)):
    #print(len(bones_3D_errors_examples))
    #print(np.array(bone_3D_std_examples[i]).shape)
    #plt.figure("std_errors_examples subject {}".format(i))
    #plt.bar(np.arange(len(bone_3D_std_examples[i])), np.mean(np.array(bone_3D_std_examples[i]), axis=(1,2)))
    plt.figure("reprojection_errors_examples subject {}".format(i))
    plt.bar(np.arange(len(reprojection_errors_examples[i])), np.min(np.array(reprojection_errors_examples[i]), axis=1))
    plt.figure("bones_3D_errors_examples subject {}".format(i))
    bottom = 0
    for j in range(len(bones_means)):
        bon_errr=np.mean(np.array(bones_3D_errors_examples[i])[:, :, j], axis=(1))/len(bones_means)*1000
        plt.bar(np.arange(len(bones_3D_errors_examples[i])), bon_errr, bottom=bottom, label="{}".format(bone_names_in_bones_list[j]))
        plt.legend()
        bottom += bon_errr
    # set y axis name
    plt.ylabel("3D bone error (mm)")
    # set x axis name
    plt.xlabel("Tasks")

    print("mean bone error subject {} : {} mm".format(i, np.mean(np.array(bones_3D_errors_examples[i]))*1000))
    print("mean reproj. error subject {} : {} px".format(i, np.mean(np.min(np.array(reprojection_errors_examples[i])))))
    to_print_bone.append("{:.1f}".format(np.mean(np.array(bones_3D_errors_examples[i]))*1000))
    to_print_reproj.append("{:.1f}".format(np.mean(np.min(np.array(reprojection_errors_examples[i])))))
    avg_repr_error.append(np.mean(np.min(np.array(reprojection_errors_examples[i]))))
avg_repr_error = np.mean(np.array(avg_repr_error))
print("avg_repr_error : " + str(avg_repr_error)+ " px")
to_print_reproj.append("{:.1f}".format(avg_repr_error))

plt.figure("bones_3D_errors_examples all subjects")
bottom = 0
bar_data = []
for bon_ in range(len(bones_means)):
    bar_data.append([[] for k in range(27)])
    for subj in range(len(bones_3D_errors_examples)):
        for ex in range(len(bones_3D_errors_examples[subj])):
            bar_data[bon_][ex].append(np.mean(np.array(bones_3D_errors_examples[subj])[ex, :, bon_]))

for bon_ in range(len(bones_means)):
    for ex in range(27):
        bar_data[bon_][ex] = np.mean(np.array(bar_data[bon_][ex]))

bar_data = np.array(bar_data)

for j in range(len(bones_means)):
    bon_errr = bar_data[j]/len(bones_means)*1000
    plt.bar(np.arange(bon_errr.shape[0]), bon_errr, bottom=bottom, label="{}".format(bone_names_in_bones_list[j]))
    bottom += bon_errr
# set y axis name
plt.ylabel("3D bone error (mm)", fontweight='bold', fontsize=18)
# set x axis name
plt.xlabel("Tasks", fontweight='bold', fontsize=18)
print('mean bone len error all subjects {} mm'.format(np.mean(np.array(bar_data))*1000))
to_print_bone.append("{:.1f}".format(np.mean(np.array(bar_data))*1000))
#print(np.array(reprojection_errors_examples).shape)
#print(np.min(np.array(reprojection_errors_examples), axis=1))

# generate a latex table with columns avg_repr_error, avg_bone_len_error
# and rows: all subjects, subject 1, subject 2, ...

to_print = ["16 views"] + to_print_bone +[" "]+ to_print_reproj

print(" & ".join(to_print))



#split the legend in 2 columns
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), ncol=2)

#plt.legend()
plt.show()