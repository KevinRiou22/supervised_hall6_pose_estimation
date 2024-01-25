import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Poses visualization script')
# General arguments
parser.add_argument('--path', help="path to logged examples", default='./')
parser.add_argument('--epoch', type=int,  help='epoch of the results to visualize')
parser.add_argument('--trial', type=int,  help='if hyperparam search, provide the trial ID', default=-1)


args = parser.parse_args()
epoch = args.epoch
path = args.path
trial_id=args.trial

if trial_id ==-1:
    absolute_path_pred = np.load(path+"absolute_path_pred"+"_epoch_" + str(epoch)+".npy")
    absolute_path_gt = np.load(path+"absolute_path_gt" + "_epoch_" + str(epoch)+".npy")
else:
    absolute_path_pred = np.load(path + "absolute_path_pred" + "_epoch_" + str(epoch) + "_trial_" + str(trial_id)+ ".npy")
    absolute_path_gt = np.load(path + "absolute_path_gt" + "_epoch_" + str(epoch) + "_trial_" + str(trial_id)+ ".npy")

for v in range(4):
    origin_pred = absolute_path_pred[0, 0, 0, 0, v]
    x_out = absolute_path_pred[:100, 0, 0, 0, v]-origin_pred
    y_out = absolute_path_pred[:100, 0, 0, 1, v]-origin_pred
    z_out = absolute_path_pred[:100, 0, 0, 2, v]-origin_pred

    origin_gt = absolute_path_gt[0, 0, 0, 0, v]
    x_gt = absolute_path_gt[:100, 0, 0, 0, v]-origin_gt
    y_gt = absolute_path_gt[:100, 0, 0, 1, v]-origin_gt
    z_gt = absolute_path_gt[:100, 0, 0, 2, v]-origin_gt

    fig = plt.figure("view_3d_" + str(v) + "_epoch_" + str(epoch))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
    ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
    ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
    # plt.savefig("view_3d_" + str(v)+"_epoch_" + str(epoch))
    plt.legend()
plt.show()