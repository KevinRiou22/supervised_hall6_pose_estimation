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
import time
set_seed()
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)
 
class ChunkedGenerator(Dataset):
    def __init__(self, batch_size, poses_2d,
                 chunk_length, camera_param = None, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False,  step = 1, sub_act = None, extra_poses_3d = None):
        tmp = []
        num_cam = len(poses_2d)  #len(poses_2d)=4; len(poses_2d[0])=150
        #len(sub_act) = 150
        self.VIEWS = range(num_cam)
        self.frame_number = None
        # print('FLAG')
        # print(len(poses_2d[0]))
        for i in range(len(poses_2d[0])): #num of videos
            n_frames = 10000000000
            for n in range(num_cam): #num of cams
                if poses_2d[n][i].shape[0] < n_frames:
                    n_frames = poses_2d[n][i].shape[0]
                
            for n in range(num_cam):
                poses_2d[n][i] = poses_2d[n][i][:n_frames]
                
            ##将多个相机视角下的二维姿态数据拼接成一个三维张量    
            temp_pos = poses_2d[0][i][..., np.newaxis]  #第0个view的第i段视频
            
            for j in range(1, num_cam):
                temp_pos = np.concatenate((temp_pos, poses_2d[j][i][...,np.newaxis]), axis = -1)

            tmp.append(temp_pos)
        self.frame_number = n_frames
        self.db = tmp  #len(self.db)=150; len(self.db[0])=2478; len(self.db[0][0])=17; len(self.db[0][0][0])=8; len(self.db[0][0][0][0])=4
        
        #print(len(self.db), len(self.db[0]))
        self.sub_act = sub_act
        self.extra_poses_3d = extra_poses_3d
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples

        for i in range(len(poses_2d[0])):#num of videos
            n_chunks = (poses_2d[0][i].shape[0] + chunk_length - 1) // chunk_length ##CHUNK_LENGTH=1, n_chunks=611
            # print('FLAG')
            # print(chunk_length, n_chunks)
            #sub_act_crt = self.sub_act[i] if self.sub_act is not None else None
            sub_act_crt = None
            offset = (n_chunks * chunk_length - poses_2d[0][i].shape[0]) // 2
            # print('FLAG')
            # print(offset)
            bounds = np.arange(n_chunks+1)*chunk_length - offset ##被分成了n_chunks块，因此总的边界是n_chunks+1
            # print('FLAG')
            # print(bounds)
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

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size ##确保每个batch的大小都是batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad ###3
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        self.current_id = []

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0][0].shape[-2], poses_2d[0][0].shape[-1], num_cam)) #16,7,17,8,4
        # if extra_poses_3d is not None:
        #     self.batch_3d = np.empty((batch_size, chunk_length + 2*pad, extra_poses_3d[0].shape[-2], extra_poses_3d[0].shape[-1]))
        # else:
        #     None
        self.batch_flip = [False for _ in range(batch_size)]
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
                pairs = self.random.permutation(self.pairs) ##使得每个batch中的数据都是随机的
                # print(pairs)
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
                # print('flag')
                # print(chunks)
                # 1个chunk包含了16个pair，顺序随机
                
                # if self.sub_act is None:
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks): ##start_3d是区间头，end_3d是区间尾，(0,1)(1,2)(2,3)...
                    self.current_id.append([seq_i, start_3d])
                    
                    start_2d = start_3d - self.pad - self.causal_shift ##casual_shift=0
                    end_2d = end_3d + self.pad - self.causal_shift
                    # print('HIIII',start_2d, end_2d)
                    # 2D poses
                    seq_2d = self.db[seq_i] ##seq_i代表这是哪一个视频 (611, 17, 8, 4)
                    # print('seq_i',seq_i)
                    # time.sleep(10)
                    #print('flag', seq_2d.shape)
                    # seq_3d = self.extra_poses_3d[seq_i] if self.extra_poses_3d is not None else None
                    self.label_sub_act[i] = self.sub_act[seq_i] if self.sub_act is not None else None
                    low_2d = max(start_2d, 0) #0 is the start frame
                    high_2d = min(end_2d, seq_2d.shape[0]) #seq_2d.shape[0] is the end frame
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    

                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                        # if start_3d == 0: #(4, 17, 8, 4)
                        #     print('seq_2d',seq_2d[low_2d:high_2d].shape)
                        #     time.sleep(10)
                        # if self.extra_poses_3d is not None:
                        #     self.batch_3d[i] = np.pad(seq_3d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]
                        
                        # print('flag', low_2d, high_2d, self.batch_2d[i].shape, seq_2d.shape)
                        # if self.extra_poses_3d is not None:
                        #     print('self.batch_3d is Not None')
                        #     self.batch_3d[i] = seq_3d[low_2d:high_2d]
                        # else:
                        #     print('self.batch_3d is None')
                    if flip:
                        # Flip 2D keypoints
                        self.batch_flip[i] = True
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
#                 else:
#                     for i, (seq_i, start_3d, end_3d, flip, _sub_act) in enumerate(chunks):
#                         start_2d = start_3d - self.pad - self.causal_shift
#                         end_2d = end_3d + self.pad - self.causal_shift
#                         # 2D poses
#                         seq_2d = self.db[seq_i]
#                         self.label_sub_act[i] = _sub_act
#                         low_2d = max(start_2d, 0)
#                         high_2d = min(end_2d, seq_2d.shape[0])
#                         pad_left_2d = low_2d - start_2d
#                         pad_right_2d = end_2d - high_2d

#                         if pad_left_2d != 0 or pad_right_2d != 0:
#                             self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d],
#                                                       ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
#                         else:
#                             self.batch_2d[i] = seq_2d[low_2d:high_2d]

#                         if flip:
#                             # Flip 2D keypoints

#                             self.batch_2d[i, :, :, 0] *= -1
#                             self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
#                                                                                   self.kps_right + self.kps_left]
#                             if self.batch_2d.shape[-2] == 8:  # (p2d_gt, p2d_pre, p3d, vis)
#                                 self.batch_2d[i, :, :, 2] *= -1
#                                 self.batch_2d[i, :, :, 4] *= -1
#                             elif self.batch_2d.shape[-2] == 6:  # (p2d, p3d, vis)
#                                 self.batch_2d[i, :, :, 2] *= -1
#                             elif self.batch_2d.shape[-2] == 11:  # (p2d_gt, p2d_pre, p3d, trj_c3d, vis)
#                                 self.batch_2d[i, :, :, 2] *= -1
#                                 self.batch_2d[i, :, :, 4] *= -1
#                                 self.batch_2d[i, :, :, 7] *= -1
#                             elif self.batch_2d.shape[-2] == 13:  # (p2d_gt, p2d_pre, p3d, trj_c3d, trj_2d, vis)
#                                 self.batch_2d[i, :, :, 2] *= -1
#                                 self.batch_2d[i, :, :, 4] *= -1
#                                 self.batch_2d[i, :, :, 7] *= -1
#                                 self.batch_2d[i, :, :, 10] *= -1
#                             else:
#                                 print(self.batch_2d.shape[-2])
#                                 sys.exit()


                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                if self.sub_act is not None:
                    #print('my current id',self.current_id)
                    
                    print('self.batch_2d.shape: {}; Length of self.label_sub_act is {}'.format(self.batch_2d.shape, len(self.label_sub_act)))
                    #time.sleep(5)
                    #print(self.label_sub_act)
                    yield self.batch_2d[:len(chunks)], self.label_sub_act[:len(chunks)], self.batch_flip[:len(chunks)], self.current_id[b_i*self.batch_size : (b_i+1)*self.batch_size], self.frame_number
                    
                else:
                    # if self.extra_poses_3d is not None:
                    #     yield self.batch_2d[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_flip[:len(chunks)]
                    # else:
                    yield self.batch_2d[:len(chunks)], self.batch_flip[:len(chunks)], None
            if self.endless:
                self.state = None
            else:
                enabled = False
