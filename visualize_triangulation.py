import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
from mpl_toolkits.mplot3d import Axes3D

path_dataset = "./data/"
path_meta_data = "./data/"
#operators = [1]

operators = [5]
tasks= [1]
examples = [2]#list(range(1, 31))

my_data_pth = path_dataset+"triangulated_3D_16_cams.npz"#"triangulated_3D_16_cams.npz"#'hall6.npz' ####one example:t1_o1_ex7
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
frame = 150

op5_task1_ex2 = []
op1_task1_ex1 = []

#enumerate data_npy, keys and values
for i, (k_s, v_s) in enumerate(data_npy.items()):
    print(k_s)
    for j, (k_a, v_a) in enumerate(v_s.item().items()):
        k_a = k_a.split("_")
        operator_str = "operator" + (''.join(filter(str.isdigit, k_s)))

        cams_videos = []

        if k_a[1]!="example2" or k_a[0]!="task1" or operator_str!="operator5":
            continue

        for cam in cams:
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
        for v in range(len(v_a)):
            print(v_a[v].shape)
            #print(v_a[v][0,:, :2])
            #print(v_a[v][0, :, 2:4])
            axs[i // 4, i % 4].set_title(cams[v]+" / "+str(v))
            first_frame = cams_videos[i][0]
            # plot first frame in subplot i
            axs[i // 4, i % 4].imshow(first_frame)


            h_inp_2d = v_a[v][frame,:, 2]
            w_inp_2d = v_a[v][frame, :, 3]


            h_3D_to_2D = v_a[v][frame, :, 0]
            w_3D_to_2D = v_a[v][frame, :, 1]

            if "r" in cams[v] or "l" in cams[v]:
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

            x_out = v_a[v][frame, :, 4]
            y_out = v_a[v][frame, :, 5]
            z_out = v_a[v][frame, :, 6]
            figs_3D.append(plt.figure("3D view {}".format(v)))
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