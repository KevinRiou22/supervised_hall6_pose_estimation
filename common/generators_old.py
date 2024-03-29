from itertools import zip_longest
import numpy as np
import torch
from torch.utils.data import Dataset
import sys,os
import copy
import random
import pickle
from common.camera import *
from common.set_seed import *
import itertools
set_seed()
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)
 
class ChunkedGenerator(Dataset):
    def __init__(self, batch_size, poses_2d,
                 chunk_length, camera_param = None, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False,  step = 1, sub_act = None, extra_poses_3d=None):
        tmp = []
        num_cam = len(poses_2d)  #len(poses_2d)=4; len(poses_2d[0])=150
        #len(sub_act) = 150
        self.VIEWS = range(num_cam)
 
        for i in range(len(poses_2d[0])): #num of videos
            n_frames = 10000000000
            for n in range(num_cam): #num of cams
                if poses_2d[n][i].shape[0] < n_frames:
                    n_frames = poses_2d[n][i].shape[0]
                
            for n in range(num_cam):
                poses_2d[n][i] = poses_2d[n][i][:n_frames]
                
            temp_pos = poses_2d[0][i][..., np.newaxis]  #第0个view的第i段视频
            for j in range(1, num_cam):
                temp_pos = np.concatenate((temp_pos, poses_2d[j][i][...,np.newaxis]), axis = -1)

            tmp.append(temp_pos)
        self.db = tmp  #len(self.db)=150; len(self.db[0])=2478; len(self.db[0][0])=17; len(self.db[0][0][0])=8; len(self.db[0][0][0][0])=4
        self.sub_act = sub_act
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples

        for i in range(len(poses_2d[0])):#num of videos
            n_chunks = (poses_2d[0][i].shape[0] + chunk_length - 1) // chunk_length
            sub_act_crt = self.sub_act[i] if self.sub_act is not None else None
            offset = (n_chunks * chunk_length - poses_2d[0][i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)

            if sub_act_crt is not None:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector, [tuple(sub_act_crt)]*len(bounds - 1))
            else:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                if sub_act_crt is not None:
                    pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector, [tuple(sub_act_crt)]*len(bounds - 1))
                else:
                    pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)
        pairs = pairs[::step]
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0][0].shape[-2], poses_2d[0][0].shape[-1], num_cam))
        #(B, 7, 17, 8, 4)
        self.label_sub_act = np.empty((batch_size,)).tolist()
                
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            #print('***************************************************')
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size] #(B, 4) [vid_inx, frm_inx_str, frm_inx_ed, flip]
                if self.sub_act is None:
                    for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                        start_2d = start_3d - self.pad - self.causal_shift
                        end_2d = end_3d + self.pad - self.causal_shift
                        # 2D poses
                        seq_2d = self.db[seq_i]

                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, seq_2d.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d

                        if pad_left_2d != 0 or pad_right_2d != 0:
                            self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_2d[i] = seq_2d[low_2d:high_2d]

                        if flip:
                            # Flip 2D keypoints

                            self.batch_2d[i, :, :, 0] *= -1
                            self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]
                            if self.batch_2d.shape[-2] == 8:#(p2d_gt, p2d_pre, p3d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                                self.batch_2d[i, :, :, 4] *= -1
                            elif self.batch_2d.shape[-2] == 6:#(p2d, p3d, vis)
                                self.batch_2d[i, :,:,2] *= -1
                            elif self.batch_2d.shape[-2] == 11: #(p2d_gt, p2d_pre, p3d, trj_c3d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                                self.batch_2d[i, :, :, 4] *= -1
                                self.batch_2d[i, :, :, 7] *= -1
                            elif self.batch_2d.shape[-2] == 13: #(p2d_gt, p2d_pre, p3d, trj_c3d, trj_2d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                                self.batch_2d[i, :, :, 4] *= -1
                                self.batch_2d[i, :, :, 7] *= -1
                                self.batch_2d[i, :, :, 10] *= -1
                            else:
                                print(self.batch_2d.shape[-2])
                                sys.exit()
                else:
                    for i, (seq_i, start_3d, end_3d, flip, _sub_act) in enumerate(chunks):
                        start_2d = start_3d - self.pad - self.causal_shift
                        end_2d = end_3d + self.pad - self.causal_shift
                        # 2D poses
                        seq_2d = self.db[seq_i]
                        self.label_sub_act[i] = _sub_act
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, seq_2d.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d

                        if pad_left_2d != 0 or pad_right_2d != 0:
                            self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d],
                                                      ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_2d[i] = seq_2d[low_2d:high_2d]

                        if flip:
                            # Flip 2D keypoints

                            self.batch_2d[i, :, :, 0] *= -1
                            self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
                                                                                  self.kps_right + self.kps_left]
                            if self.batch_2d.shape[-2] == 8:  # (p2d_gt, p2d_pre, p3d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                                self.batch_2d[i, :, :, 4] *= -1
                            elif self.batch_2d.shape[-2] == 6:  # (p2d, p3d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                            elif self.batch_2d.shape[-2] == 11:  # (p2d_gt, p2d_pre, p3d, trj_c3d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                                self.batch_2d[i, :, :, 4] *= -1
                                self.batch_2d[i, :, :, 7] *= -1
                            elif self.batch_2d.shape[-2] == 13:  # (p2d_gt, p2d_pre, p3d, trj_c3d, trj_2d, vis)
                                self.batch_2d[i, :, :, 2] *= -1
                                self.batch_2d[i, :, :, 4] *= -1
                                self.batch_2d[i, :, :, 7] *= -1
                                self.batch_2d[i, :, :, 10] *= -1
                            else:
                                print(self.batch_2d.shape[-2])
                                sys.exit()


                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                if self.sub_act is not None:
                    yield self.batch_2d[:len(chunks)], self.label_sub_act
                else:
                    yield self.batch_2d[:len(chunks)], None
            if self.endless:
                self.state = None
            else:
                enabled = False
