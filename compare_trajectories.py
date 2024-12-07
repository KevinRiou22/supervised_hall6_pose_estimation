import numpy as np
import argparse
import matplotlib.pyplot as plt
from pynput import keyboard
import torch
import tkinter as tk
from tkinter import simpledialog
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def plot_one_pose(pose):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
    plt.show()

def plot_all_video_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for pose in poses:
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
    plt.show()

def on_press(key):
    try:
        if key.char == 'q':
            i+=1
        elif key.char == 'a':
            i-=1
    except AttributeError:
        pass

def on_release(key):
    pass


def player_poses_2_traj(poses_ref, poses):
    """ if q is pressed, display next pose, 
    if  a is pressed, display previous pose"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #make the plot dynamic
    plt.ion()
    i=0
    max_len = max(poses_ref.shape[0], poses.shape[0])
    while i<max_len:
        pose_ref = poses_ref[i]
        pose = poses[i]
        ax.clear()
        #fix the space to -2, 2 in x and y axis and 0, 4 in z axis
        #display the frame as titel
        ax.set_title(f"Frame: {i}")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 4)
        #plot axis names
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if i<poses_ref.shape[0]:
            ax.scatter(pose_ref[:, 0], pose_ref[:, 1], pose_ref[:, 2])
        if i<poses.shape[0]:
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
        plt.pause(0.1)
        i+=1

        #wait for key press 
        # with keyboard.Events() as events:
        #     for event in events:
        #         print(event.key)
        #         # if q is pressed, display next pose
        #         if event.key == keyboard.KeyCode.from_char('q'):
        #             i+=1
        #             print("next")
        #             break

def play_traj_with_closest_ref(poses_ref, poses, pairs):
    """ if q is pressed, display next pose, 
    if  a is pressed, display previous pose"""
    bones = [[11, 12], [12, 14], [14, 16], [11, 13], [13, 15], [5, 7], [7, 9], [6, 8], [8, 10], [4,2], [3,1], [2,0], [1,0]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #make the plot dynamic
    plt.ion()
    i=0
    max_len = max(poses_ref.shape[0], poses.shape[0])
    while i<max_len:
        # pose_ref = poses_ref[i]
        pose = poses[i]
        #search pair among pairs where pair[1] == i
        pair = [pair for pair in pairs if pair[1] == i]
        if len(pair) == 0:
            print("No pair found")
            break
        pair = pair[0]
        pose_ref = poses_ref[pair[0]]


        ax.clear()
        #fix the space to -2, 2 in x and y axis and 0, 4 in z axis
        #display the frame as titel
        ax.set_title(f"Frame: {i}")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 4)
        #plot axis names
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if i<poses_ref.shape[0]:
            ax.scatter(pose_ref[:, 0], pose_ref[:, 1], pose_ref[:, 2], color='blue', label='ref')
        if i<poses.shape[0]:
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color='red', label='target')
        #draw the bones
        for bone in bones:
            ax.plot([pose_ref[bone[0], 0], pose_ref[bone[1], 0]], [pose_ref[bone[0], 1], pose_ref[bone[1], 1]], [pose_ref[bone[0], 2], pose_ref[bone[1], 2]], color='blue')
            ax.plot([pose[bone[0], 0], pose[bone[1], 0]], [pose[bone[0], 1], pose[bone[1], 1]], [pose[bone[0], 2], pose[bone[1], 2]], color='red')
        #legend
        ax.legend()
        plt.pause(0.1)
        i+=1

        #wait for key press 
        # with keyboard.Events() as events:
        #     for event in events:
        #         print(event.key)
        #         # if q is pressed, display next pose
        #         if event.key == keyboard.KeyCode.from_char('q'):
        #             i+=1
        #             print("next")
        #             break

def align_traj_based_on_ref(traj_ref, traj):
    traj_ref = torch.tensor(traj_ref)
    traj = torch.tensor(traj)
    traj_ref = traj_ref.unsqueeze(0)
    traj = traj.unsqueeze(0)



    muX = torch.mean(traj_ref[:, 0], dim=1, keepdims=True)
    muY = torch.mean(traj[:, 0], dim=1, keepdims=True)
    
    X0 = traj_ref[:,0] - muX
    Y0 = traj[:, 0] - muY

    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    
    X0 = X0/normX
    Y0 = Y0/normY

    H = torch.matmul(X0.permute(0, 2, 1), Y0)
    U, s, Vt = torch.linalg.svd(H)
    U_ = U.permute(0, 2, 1).contiguous()
    V = Vt.permute(0, 2, 1).contiguous()
    R = torch.matmul(V, U_)

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = torch.sign(torch.unsqueeze(torch.linalg.det(R), 1))
    V_2 = V.clone()
    V_2[:, :, -1] = V_2[:, :, -1].clone()*sign_detR
    sign_detR_ = sign_detR.flatten()
    s_2 = s.clone()
    s_2[:, -1] = s_2[:, -1].clone()*sign_detR_
    R_2 = torch.matmul(V_2, U_) # Rotation
    

    tr = torch.unsqueeze(torch.sum(s_2, dim=1, keepdims=True), 2)

    a = tr * normX / normY # Scale
    t = muX - a*torch.matmul(muY, R_2) # Translation
    print(a.shape, t.shape, R_2.shape)
    # Perform rigid transformation on the input
    for i in range(traj.shape[1]):
        traj[:, i] = torch.matmul(traj[:, i], R_2)
        traj[:, i] = a * traj[:, i]
        traj[:, i] = traj[:, i] + t
    return traj.numpy().squeeze()
    



def save_labels(labels, path):
    with open(path, 'w') as f:
        json.dump(labels, f)

def save_un_annotated_actions(actions, path):
    with open(path, 'w') as f:
        json.dump(actions, f)





path_dataset = "/home/kevin-riou/Documents/supervised_hall6_pose_estimation/data/"
file_name = "data_epoch_59_16_cams.npz" # "triangulate_16_cams.npz" #"data_epoch_59_16_cams.npz"
my_data_pth = path_dataset+ file_name
my_data = np.load(my_data_pth, allow_pickle=True)
my_data = dict(my_data)

S1 = my_data['S1'].item()
S2 = my_data['S2'].item()
S3 = my_data['S3'].item()
S4 = my_data['S4'].item()
S5 = my_data['S5'].item()
task = 1
example = 1
print(len(S1.keys()))
print(S1.keys())
input()
traj_ref = S2['task{}_example{}'.format(task, example)][0][:400, :, 4:7]
example = 2 #crouching vs bending
traj = S2['task{}_example{}'.format(task, example)][0][:400, :, 4:7]

traj = align_traj_based_on_ref(traj_ref, traj)

traj_ref_flat = traj_ref-traj_ref[:, 0:1]
traj_flat = traj-traj[:, 0:1]

traj_ref_flat = traj_ref.reshape(traj_ref_flat.shape[0], -1)
traj_flat = traj.reshape(traj_flat.shape[0], -1)

distance, path = fastdtw(traj_ref_flat, traj_flat, dist=euclidean)

print(path)
print(distance)


play_traj_with_closest_ref(traj_ref, traj, path)


# print(traj_ref.shape)
# # plot_one_pose(traj_ref[0])
# # player_poses(traj_ref)
# 

# player_poses_2_traj(traj_ref, traj)