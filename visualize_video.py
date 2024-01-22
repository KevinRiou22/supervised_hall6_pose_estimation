import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

def main(operator, task, example):
    # print("Operator: ", operator)
    # print("Task: ", task)
    # print("Example: ", example)
    # load videos from data/images/task/operator/example using opencv :
    cams_videos = []
    #create a 4*3 subplot

    frame_range = [150, 160]
    cams = os.listdir('data/images/task{}/operator{}/example{}'.format(task, operator, example))
    for cam in cams:
        # print(cam)
        cams_videos.append(read_video_cv2('data/images/task{}/operator{}/example{}/{}/video.avi'.format(task, operator, example, cam), frame_range))

    for t in range((frame_range[1]-frame_range[0])):
        fig, axs = plt.subplots(4, 4)
        # remove x and y ticks and labels
        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])
        i = 0
        for cam in cams:
            # set subtitle for subplot i with cam
            axs[i // 4, i % 4].set_title(cam)
            first_frame = cams_videos[i][t]
            # plot first frame in subplot i
            axs[i // 4, i % 4].imshow(first_frame)
            i += 1

        plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, help='task id', required=True)
    parser.add_argument('--operator', type=int, help='operator id', required=True)
    parser.add_argument('--example', type=str, help='example_id', required=True)

    args = parser.parse_args()

    main(args.operator, args.task, args.example)