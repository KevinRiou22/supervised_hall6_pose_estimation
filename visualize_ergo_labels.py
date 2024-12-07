import numpy as np
import argparse
import matplotlib.pyplot as plt
from pynput import keyboard
import torch
import tkinter as tk
from tkinter import simpledialog
import json
import threading
from queue import Queue

def player_poses(poses, state, path_labels_file, path_un_annotated_actions, un_annotated_actions, all_actions, ergo_variations):
    """
    Display 3D poses dynamically 
    - Press right arrow to display the next pose.
    - Press left arrow to display the previous pose.
    - Press 'esc' to exit.
    """

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                state['index'] = (state['index'] + 1) % poses.shape[0] 
                if state['labels'].get(str(state['index'])) is not None :
                    if state['labels'].get(str(state['index'])).startswith("start"):
                        state['action_counter'] += 1
            elif key == keyboard.Key.left:
                state['index'] = (state['index'] - 1) % poses.shape[0]
                if state['labels'].get(str(state['index']+1)) is not None :
                    if state['labels'].get(str(state['index']+1)).startswith("start"):
                        state['action_counter'] -= 1
            elif key == keyboard.Key.esc:
                state['running'] = False
                return False
        except AttributeError:
            if key == keyboard.Key.esc:  # Stop the loop on 'esc'
                state['running'] = False
                return False  # Stop the listener

    # Listener in a separate thread
    def start_listener():
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    # Disable key listening for the matplotlib figure
    fig = plt.figure()
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    ax = fig.add_subplot(projection='3d')
    plt.ion()  # Enable interactive mode
    if state['labels'].get(str(state['index'])) is not None :
        if state['labels'].get(str(state['index'])).startswith("start"):
            state['action_counter'] += 1
    current_label = state['labels'].get(str(state['index']), "No label")
    current_ergo_label = ergo_variations[state['action_counter']] if state['action_counter'] != -1 else "No Ergo label"
    ax.set_title(f"Frame: {state['index']} | Label: {current_label} | Ergo Label: {current_ergo_label}")
    # display the list of actions below the plot with put text
    print("Un-annotated actions: ", un_annotated_actions)

    try:
        while state['running']:
            ax.clear()  # Clear the plot
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(0, 4)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Plot the current pose
            pose = poses[state['index']]
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
            # Display the current label if available
            current_label = state['labels'].get(str(state['index']), "No label")
            current_ergo_label = ergo_variations[state['action_counter']] if state['action_counter'] != -1 else "No Ergo label"
            ax.set_title(f"Frame: {state['index']} | Label: {current_label} | Ergo Label: {current_ergo_label}")
            plt.pause(0.1)

    finally:
        plt.ioff()  # Turn off interactive mode
        listener_thread.join()  # Ensure the listener thread stops



def main(args):
    task_actions = {1:["start_take_from_shelve", "end_take_from_shelve", "start_transport", "end_transport", "start_put_on_table", "end_put_on_table", "start_take_from_table", "end_take_from_table", "start_transport", "end_transport", "start_put_on_shelve", "end_put_on_shelve"],
                    2:["start_walk", "end_walk", "start_take_from_table", "end_take_from_table", "start_screw", "end_screw", "start_put_on_table", "end_put_on_table", "start_walk", "end_walk"],
                    3:["start_walk", "end_walk", "start_take_from_table", "end_take_from_table", "start_hammer", "end_hammer", "start_put_on_table", "end_put_on_table", "start_walk", "end_walk"]
                    }
    all_actions = task_actions[args.task]
    # Load the data
    data = np.load(args.path_poses + args.poses_filename, allow_pickle=True)
    data = dict(data)
    subject = data['S{}'.format(args.subject)].item()
    poses = subject['task{}_example{}'.format(args.task, args.example)][0][:, :, 4:7]
    path_labels_file = args.path_labels + 'S{}_task{}_example{}_labels.json'.format(args.subject, args.task, args.example)
    path_un_annotated_actions = args.path_labels + 'S{}_task{}_example{}_un_annotated_actions.json'.format(args.subject, args.task, args.example)
    #check if their's a json file 'S{}_task{}_example{}_labels.json' in the path_labels
    try:
        with open(path_labels_file, 'r') as f:
            labels = json.load(f)
        with open(path_un_annotated_actions, 'r') as f:
            un_annotated_actions = json.load(f)
    except:
        print("No labels found for this subject/task/example yet")
    state = {
        'index': 0,
        'running': True,
        'labels': labels,
        'max_len': poses.shape[0],
        'action_counter': -1
    }

    task_ergo_label_variations ={"T1":{"E1":["crouch", "chest", "rotate", "rotate", "chest", "crouch"],
                                    "E2":["bend", "side", "rotate", "rotate", "side", "bend"],
                                    "E3":["crouch", "chest", "rotate", "rotate", "chest", "crouch"],
                                    "E4":["bend", "side", "rotate", "rotate", "side", "bend"],
                                    "E5":["upright", "chest", "rotate", "rotate", "chest", "upright"],
                                    "E6":["bend", "side", "rotate", "rotate", "side", "bend"],
                                    "E7":["upright", "chest", "rotate", "rotate", "chest", "upright"],
                                    "E8":["upright", "chest", "rotate", "rotate", "chest", "upright"],
                                    "E9":["upright", "side", "rotate", "rotate", "side", "upright"],
                                    "E10":["random", "shoulder", "rotate", "rotate", "shoulder", "random"],
                                    "E11":["random", "shoulder", "rotate", "rotate", "shoulder", "random"],
                                    "E12":["random", "shoulder", "rotate", "rotate", "shoulder", "random"],
                                    "E13":["random", "body_far", "rotate", "rotate", "body_far", "random"],
                                    "E14":["random", "body_far", "rotate", "rotate", "body_far", "random"],
                                    "E15":["random", "body_far", "rotate", "rotate", "body_far", "random"]
                                    },
                            "T2":{"E1":["-", "left_far", "left_close", "left_far", "-"],
                                    "E2":["-", "center_far", "center_close", "center_far", "-"],
                                    "E3":["-", "right_far", "right_close", "right_far", "-"],
                                    "E4":["-", "left_close", "left_far", "left_close", "-"],
                                    "E5":["-", "center_close", "center_far", "center_close", "-"],
                                    "E6":["-", "right_close", "right_far", "right_close", "-"]
                                    },
                            "T3":{"E1":["-", "left_far", "left_close", "left_far", "-"],
                                    "E2":["-", "center_far", "center_close", "center_far", "-"], #+"wrist broken"
                                    "E3":["-", "right_far", "right_close", "right_far", "-"],
                                    "E4":["-", "left_close", "left_far", "left_close", "-"],
                                    "E5":["-", "center_close", "center_far", "center_close", "-"],
                                    "E6":["-", "right_close", "right_far", "right_close", "-"]
                                    }
                            }

    player_poses(poses, state, path_labels_file, path_un_annotated_actions, un_annotated_actions, all_actions, task_ergo_label_variations["T{}".format(args.task)]["E{}".format(args.example)])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D poses from a numpy file.")
    parser.add_argument("--path_poses", type=str, help="Path to the numpy file containing the poses.")
    parser.add_argument("--poses_filename", type=str, help="Name of the numpy file containing the poses.")
    parser.add_argument("--subject", type=int, help="Subject to visualize.")
    parser.add_argument("--task", type=int, help="Task to visualize.")
    parser.add_argument("--example", type=int, help="Example to visualize.")
    parser.add_argument("--path_labels", type=str, help="Name of the file containing the labels.")
    args = parser.parse_args()
    main(args)