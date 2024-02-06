import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from common.config import config as cfg
from common.arguments import parse_args
from common.config import reset_config, update_config
from collections import OrderedDict
import matplotlib
import cv2


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

font = {'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

args = parse_args()
update_config(args.cfg)  ###config file->cfg
reset_config(cfg, args)

parser = argparse.ArgumentParser(description='Poses visualization script')
# General arguments
#parser.add_argument('--path', help="path to logged examples", default='./')
#parser.add_argument('--epoch', type=int,  help='epoch of the results to visualize')
#parser.add_argument('--trial', type=int,  help='if hyperparam search, provide the trial ID', default=-1)

epoch = args.visu_epoch
path = args.visu_path
# trial_id=args.trial

#if trial_id ==-1:
inputs_2d_gt = np.load(path+"inputs_2d_gt"+"_epoch_" + str(epoch)+".npy")
prj_out_abs_to_2d = np.load(path+"prj_out_abs_to_2d" + "_epoch_" + str(epoch)+".npy")
inputs_3d_gt = np.load(path+"inputs_3d_gt" + "_epoch_" + str(epoch)+".npy")
out = np.load(path+"out" + "_epoch_" + str(epoch)+".npy")
# else:
#     inputs_2d_gt = np.load(path + "inputs_2d_gt" + "_epoch_" + str(epoch)+ "_trial_" + str(trial_id) + ".npy")
#     prj_out_abs_to_2d = np.load(path + "prj_out_abs_to_2d" + "_epoch_" + str(epoch)+ "_trial_" + str(trial_id) + ".npy")
#     inputs_3d_gt = np.load(path + "inputs_3d_gt" + "_epoch_" + str(epoch)+ "_trial_" + str(trial_id) + ".npy")
#     out = np.load(path + "out" + "_epoch_" + str(epoch)+ "_trial_" + str(trial_id) + ".npy")


f = open('data/bones_length_hall6_2d_pose_structure.json', )
gt_bones_lens = json.load(f)
# = cfg.H36M_DATA.BONES_NAMES
bones_h36m = cfg.H36M_DATA.BONES
bones_means_ = []
bones = []
count_subj=0
symmetry_bones = [[],[]]
f = open('data/bone_priors_mean_h36m.json', )
gt_bones_lens = json.load(f)
for k in gt_bones_lens.keys():
    gt_bones_lens[k] = np.array(gt_bones_lens[k])
bones = np.array(cfg.H36M_DATA.BONES)

f = open('data/cams_params_all_examples.json', )

params = json.load(f)
# params = dict(params)
#print(params)



pad = 7 // 2
camera_positions_origin = []
camera_positions_depth= []
for idx, view_idx in enumerate(cfg.H36M_DATA.TEST_CAMERAS):
    cam_resolution = (1000, 1000)
    fig = plt.figure("view_2d_" + str(view_idx) + "_epoch_" + str(epoch))
    print("inputs_2d_gt.shape : " + str(inputs_2d_gt.shape))
    print("prj_out_abs_to_2d : " + str(prj_out_abs_to_2d.shape))
    print("cam_resolution : " + str(cam_resolution))
    h_inp_2d = (inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, :, 0, view_idx]+1)/2*cam_resolution[0]
    w_inp_2d = (inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, :, 1, view_idx]+1)/2*cam_resolution[1]
    h_3D_to_2D = (prj_out_abs_to_2d.transpose(0, 2, 3, 4, 1)[0, 0, :, 0, idx]+1)/2*cam_resolution[0]
    w_3D_to_2D = (prj_out_abs_to_2d.transpose(0, 2, 3, 4, 1)[0, 0, :, 1, idx]+1)/2*cam_resolution[1]
    for bone_id in range(len(bones)):
        bone = bones[bone_id]
        plt.plot([h_inp_2d[bone[0]], h_inp_2d[bone[1]]], [w_inp_2d[bone[0]], w_inp_2d[bone[1]]], color='r',
                 linestyle='-', linewidth=1)
        plt.plot([h_3D_to_2D[bone[0]], h_3D_to_2D[bone[1]]], [w_3D_to_2D[bone[0]], w_3D_to_2D[bone[1]]], color='g',
                 linestyle='-', linewidth=1)

    # plt.figure("out 3D--> 2D vs input 2D, view : " + str(v))
    plt.scatter(h_inp_2d, w_inp_2d, color='r', marker='o', label="2D inputs")
    plt.scatter(h_3D_to_2D, w_3D_to_2D, color='g', marker='+', label="2D proj. from pred. 3D")
    #vid_ = read_video_cv2('data/images/{}/{}/{}/{}/video.avi'.format("task1", "operator5", "example2", cameras_intrinsic_params[view_idx]["id"]),[pad, pad+1])
    #first_frame = vid_[0]
    # plot first frame in subplot i
    #plt.imshow(first_frame)
    plt.legend()
    # plt.savefig("view_2d_" + str(v)+"_epoch_" + str(epoch))
    # plt.clf()
    root = inputs_3d_gt[:, :, :1, :, :]
    out = out - out[:, :, :1, :, :] #+ root
    out[:, :, :1, :, :] = 0
    inputs_3d_gt[:, :, :1, :, :]=0
    x_out = out[0, 0, :, 0, idx]
    y_out = out[0, 0, :, 1, idx]
    z_out = out[0, 0, :, 2, idx]

    x_gt = inputs_3d_gt[0, 0, :, 0, view_idx]
    y_gt = inputs_3d_gt[0, 0, :, 1, view_idx]
    z_gt = inputs_3d_gt[0, 0, :, 2, view_idx]

    fig = plt.figure("view_3d_" + str(view_idx) + "_epoch_" + str(epoch))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
    ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred. 3D")
    # draw 3D bones
    for bone_id in range(len(bones)):
        bone = bones[bone_id]
        ax.plot([x_gt[bone[0]], x_gt[bone[1]]], [y_gt[bone[0]], y_gt[bone[1]]], [z_gt[bone[0]], z_gt[bone[1]]],
                color='r', linestyle='-', linewidth=1)
    ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
    plt.legend()


plt.show()
