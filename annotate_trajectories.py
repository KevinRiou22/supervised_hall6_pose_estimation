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
import cv2

def player_poses(poses, state, path_labels_file, path_un_annotated_actions, un_annotated_actions, all_actions, videos, display_images):
    """
    Display 3D poses dynamically with a persistent Tkinter window for labeling.
    - Press right arrow to display the next pose.
    - Press left arrow to display the previous pose.
    - Press 'esc' to exit.
    """
    # # Shared state variables
    # state = {
    #     'index': 0,
    #     'running': True,
    #     'labels': {}  # Dictionary to store labels for each frame
    # }
    
    # Queue for communication between threads
    
    #read the videos
    def extract_images_from_video(video_path):
        cap = cv2.VideoCapture(video_path)
        images = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            #downsample the frame
            h_w_ratio = frame.shape[0]/frame.shape[1]
            target_width = 512
            target_height = int(target_width * h_w_ratio)
            frame = cv2.resize(frame, (target_width, target_height))
            images.append(frame)
        return images

    if display_images:
        videos_images = []
        for video in videos:
            videos_images.append(extract_images_from_video(video))
    

    gui_queue = Queue()

    def process_gui_queue():
        """Process messages from other threads in the Tkinter GUI loop."""
        while not gui_queue.empty():
            command, data = gui_queue.get()
            if command == "update_frame":
                update_label_display(data)
        root.after(100, process_gui_queue)

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                state['index'] = (state['index'] + 1) % poses.shape[0]
                gui_queue.put(("update_frame", state['index']))
            elif key == keyboard.Key.left:
                state['index'] = (state['index'] - 1) % poses.shape[0]
                gui_queue.put(("update_frame", state['index']))
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

    if display_images:
        fig_images = plt.figure()
        fig_images.canvas.mpl_disconnect(fig_images.canvas.manager.key_press_handler_id)
        # prepare subplots for the len(videos) videos
        axs = []
        for i in range(len(videos)):
            axs.append(fig_images.add_subplot(2, 2, i+1))

        plt.ion()  # Enable interactive mode

    # Disable key listening for the matplotlib figure
    fig = plt.figure()
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    ax = fig.add_subplot(projection='3d')
    plt.ion()  # Enable interactive mode

    # Tkinter GUI setup
    root = tk.Tk()
    root.title("Frame Labeler")

    label_display = tk.Label(root, text="Frame: 0")
    label_display.pack(pady=10)

    label_entry = tk.Entry(root, width=30)
    label_entry.pack(pady=5)

    def save_label():
        """Save the label entered in the entry field for the current frame."""
        label = label_entry.get()
        if label in all_actions:
            #if their is already a label for the current frame
            if str(state['index']) in state['labels']:
                print("A label is already assigned to this frame. Remove it first. To remove it, juste fill the label with a blank and click on 'save label' button. Note: prefer removing the start_ or end_  frame label of the action rather than intermediate label.")
                print("Also don't forget to move to next frame to add a start_ label after you labeled the previous action's end_ label! ;)")
            else:
                state['labels'][str(state['index'])] = label
                update_label_display(state['index'])
                save_labels(state['labels'], path_labels_file)
                if label.startswith("start_"):
                    un_annotated_actions.remove(label)
                    save_un_annotated_actions(un_annotated_actions, path_un_annotated_actions)
                    label = label[6:]
                    #check if the end label is already in the labels
                    if "end_"+label in state['labels'].values():
                        # tehn all the frames between the start and the end have "label" as label
                        index_end = list(state['labels'].keys())[list(state['labels'].values()).index("end_"+label)]
                        for i in range(state['index']+1, int(index_end)):
                            state['labels'][str(i)] = label
                            save_labels(state['labels'], path_labels_file)
                    print('---------------------------------------------------------')
                    print("Un-annotated actions: ", un_annotated_actions)
                if label.startswith("end_"):
                    un_annotated_actions.remove(label)
                    save_un_annotated_actions(un_annotated_actions, path_un_annotated_actions)
                    label = label[4:]
                    #check if the start label is already in the labels
                    if "start_"+label in state['labels'].values():
                        # tehn all the frames between the start and the end have "label" as label
                        index_start = list(state['labels'].keys())[list(state['labels'].values()).index("start_"+label)]
                        for i in range(int(index_start)+1, state['index']):
                            state['labels'][str(i)] = label
                            save_labels(state['labels'], path_labels_file)
                    print('---------------------------------------------------------')
                    print("Un-annotated actions: ", un_annotated_actions)
        elif label == "":
            un_annotated_actions.insert(0, state['labels'][str(state['index'])])
            save_un_annotated_actions(un_annotated_actions, path_un_annotated_actions)
            label_in_state = state['labels'].pop(str(state['index']))
            if label_in_state.startswith("start_"):
                label = label_in_state[6:]
                #check if the end label is already in the labels
                if "end_"+label in state['labels'].values():
                    # tehn all the frames between the start and the end must be poped out
                    index_end = list(state['labels'].keys())[list(state['labels'].values()).index("end_"+label)]
                    for i in range(state['index']+1, int(index_end)):
                        state['labels'].pop(str(i))
            elif label_in_state.startswith("end_"):
                label = label_in_state[4:]
                #check if the start label is already in the labels
                if "start_"+label in state['labels'].values():
                    # tehn all the frames between the start and the end must be poped out
                    index_start = list(state['labels'].keys())[list(state['labels'].values()).index("start_"+label)]
                    for i in range(int(index_start)+1, state['index']):
                        state['labels'].pop(str(i))
            save_labels(state['labels'], path_labels_file)
            print('---------------------------------------------------------')
            print("Un-annotated actions: ", un_annotated_actions)     
        else:
            print("Invalid label. Please enter a valid label.")

    def update_label_display(frame_index):
        """Update the label display with the current frame and label."""
        current_label = state['labels'].get(frame_index, "No label")
        label_display.config(text=f"Frame: {frame_index} ")

    save_button = tk.Button(root, text="Save Label", command=save_label)
    save_button.pack(pady=10)

    # Schedule GUI queue processing
    root.after(100, process_gui_queue)

    # Run the GUI in the main thread
    threading.Thread(target=root.mainloop, daemon=True).start()

    update_label_display(state['index'])
    current_label = state['labels'].get(str(state['index']), "No label")
    ax.set_title(f"Frame: {state['index']} | Label: {current_label}")
    if  display_images:
        for i in range(len(videos)):
            axs[i].imshow(videos_images[i][state['index']])

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
            if display_images:
                for i in range(len(videos)):
                    axs[i].imshow(videos_images[i][state['index']])
            # Display the current label if available
            current_label = state['labels'].get(str(state['index']), "No label")
            ax.set_title(f"Frame: {state['index']} | Label: {current_label}")

            plt.pause(0.1)

    finally:
        plt.ioff()  # Turn off interactive mode
        listener_thread.join()  # Ensure the listener thread stops

        return state['labels']


def save_labels(labels, path):
    with open(path, 'w') as f:
        json.dump(labels, f)

def save_un_annotated_actions(actions, path):
    with open(path, 'w') as f:
        json.dump(actions, f)

def main(args):
    task_actions = {1:["start_take_from_shelve", "end_take_from_shelve", "start_transport_1", "end_transport_1", "start_put_on_table", "end_put_on_table", "start_take_from_table", "end_take_from_table", "start_transport_2", "end_transport_2", "start_put_on_shelve", "end_put_on_shelve"],
                    2:["start_walk_1", "end_walk_1", "start_take_from_table", "end_take_from_table", "start_screw", "end_screw", "start_put_on_table", "end_put_on_table", "start_walk_2", "end_walk_2"],
                    3:["start_walk_1", "end_walk_1", "start_take_from_table", "end_take_from_table", "start_hammer", "end_hammer", "start_put_on_table", "end_put_on_table", "start_walk_2", "end_walk_2"]
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
        labels = {}
        un_annotated_actions = all_actions
    state = {
        'index': 0,
        'running': True,
        'labels': labels,
        'max_len': poses.shape[0]
    }
    video_1_path = args.path_poses + "images/task{}/operator{}/example{}/l_10028260/video.avi".format(args.task, args.subject, args.task, args.example)
    video_2_path = args.path_poses + "images/task{}/operator{}/example{}/l_10028261/video.avi".format(args.task, args.subject, args.task, args.example)
    video_3_path = args.path_poses + "images/task{}/operator{}/example{}/l_10028262/video.avi".format(args.task, args.subject, args.task, args.example)
    video_4_path = args.path_poses + "images/task{}/operator{}/example{}/l_10028263/video.avi".format(args.task, args.subject, args.task, args.example)
    videos = [video_1_path, video_2_path, video_3_path, video_4_path]
    labels = player_poses(poses, state, path_labels_file, path_un_annotated_actions, un_annotated_actions, all_actions, videos, args.display_images)
    save_labels(labels, path_labels_file)
    save_un_annotated_actions(un_annotated_actions, path_un_annotated_actions)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D poses from a numpy file.")
    parser.add_argument("--path_poses", type=str, help="Path to the numpy file containing the poses.")
    parser.add_argument("--poses_filename", type=str, help="Name of the numpy file containing the poses.")
    parser.add_argument("--subject", type=int, help="Subject to visualize.")
    parser.add_argument("--task", type=int, help="Task to visualize.")
    parser.add_argument("--example", type=int, help="Example to visualize.")
    parser.add_argument("--path_labels", type=str, help="Name of the file containing the labels.")
    parser.add_argument("--display_images", type=bool, default=False, help="Display the images of the videos.")
    args = parser.parse_args()
    main(args)