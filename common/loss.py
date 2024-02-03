from os import device_encoding
import torch
import numpy as np
import math 
import sys
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch.nn.functional as F
#from common.svd.batch_svd import batch_svd
def eval_metrc(cfg, predicted, target):
    '''
    predicted:(B, T, J, C)
    '''
    B, T, J, C = predicted.shape
    if cfg.TEST.METRIC == 'mpjpe':
        eval_loss = mpjpe(predicted, target)
    elif cfg.TEST.METRIC == 'p_mpjpe':
        predicted = predicted.view(B*T, J, C)
        target = target.view(B*T, J, C)
        eval_loss = p_mpjpe(cfg, predicted, target)
    elif cfg.TEST.METRIC == 'n_mpjpe':
        eval_loss = n_mpjpe(predicted, target)

    return eval_loss


def get_mat_torch(x, y): 
    z = torch.cross(y, x)
    y = torch.cross(x, z)
    x = torch.cross(z, y)
    mat = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)
    return mat 


def get_poses_torch(joints): 

    # input joints expected to be (N, 17, 3)
    # the parent and child link of some joint 
    parents, children = [], []
    # r knee 
    xp = torch.cross(joints[:, 1] - joints[:, 2], joints[:, 3] - joints[:, 2])
    yp = joints[:, 2] - joints[:, 1]
    xc = torch.cross(joints[:, 1] - joints[:, 2], joints[:, 3] - joints[:, 2])
    yc = joints[:, 3] - joints[:, 2]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # r hip 
    xp = joints[:, 1] - joints[:, 4]
    yp = joints[:, 0] - joints[:, 7]
    xc = torch.cross(joints[:, 1] - joints[:, 2], joints[:, 3] - joints[:, 2])
    yc = joints[:, 2] - joints[:, 1]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # l hip 
    xp = joints[:, 1] - joints[:, 4]
    yp = joints[:, 0] - joints[:, 7]
    xc = torch.cross(joints[:, 4] - joints[:, 5], joints[:, 6] - joints[:, 5])
    yc = joints[:, 5] - joints[:, 4]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # l knee 
    xp = torch.cross(joints[:, 4] - joints[:, 5], joints[:, 6] - joints[:, 5])
    yp = joints[:, 5] - joints[:, 4]
    xc = torch.cross(joints[:, 4] - joints[:, 5], joints[:, 6] - joints[:, 5])
    yc = joints[:, 6] - joints[:, 5]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # re 
    xp = joints[:, 15] - joints[:, 14]
    yp = torch.cross(joints[:, 14] - joints[:, 15], joints[:, 16] - joints[:, 15])
    xc = joints[:, 16] - joints[:, 15]
    yc = torch.cross(joints[:, 14] - joints[:, 15], joints[:, 16] - joints[:, 15])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # rs 
    xp = joints[:, 14] - joints[:, 11]
    yp = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    xc = joints[:, 15] - joints[:, 14]
    yc = torch.cross(joints[:, 14] - joints[:, 15], joints[:, 16] - joints[:, 15])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # ls 
    xp = joints[:, 14] - joints[:, 11]
    yp = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    xc = joints[:, 11] - joints[:, 12]
    yc = torch.cross(joints[:, 13] - joints[:, 12], joints[:, 11] - joints[:, 12])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # le 
    xp = joints[:, 11] - joints[:, 12]
    yp = torch.cross(joints[:, 13] - joints[:, 12], joints[:, 11] - joints[:, 12])
    xc = joints[:, 12] - joints[:, 13]
    yc = torch.cross(joints[:, 13] - joints[:, 12], joints[:, 11] - joints[:, 12])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # thorax 
    xp = joints[:, 1] - joints[:, 4]
    yp = joints[:, 0] - joints[:, 7]
    xc = joints[:, 14] - joints[:, 11]
    yc = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # pelvis 
    
    xc = joints[:, 1] - joints[:, 4]
    yc = ((joints[:, 1] + joints[:, 4]) / 2 - joints[:, 0]) * 100.0 
    mask = torch.nonzero((((joints[:, 1] + joints[:, 4]) / 2 - joints[:, 0]) * 100.0).sum(dim=-1) < 10.0)
    if len(mask) > 0: 
        mask = mask.squeeze()
        yc[mask] = joints[mask, 0] - joints[mask, 7]

    c = get_mat_torch(xc, yc)
    children.append(c.unsqueeze(1))

    # neck 8
    xp = joints[:, 14] - joints[:, 11]
    yp = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    xc = torch.cross(joints[:, 8] - joints[:, 10], joints[:, 9] - joints[:, 10])
    yc = (joints[:, 11] + joints[:, 14]) / 2 - joints[:, 10]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    
    #[r knee, r hip, l hip, l knee, re, rs, ls, le, thorax, pelvis, neck 8]
    # concat -> out shape: (N, 11, 3, 3)
    N = len(parents)
    parents = torch.cat(parents, dim=1)
    children = torch.cat(children, dim=1)
    
    
    # normalize 
    parents = parents / (parents.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-8)
    children = children / (children.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-8)

 
    return parents, children
def align_numpy(source, target):
    '''
    Args:
        source : (B, J, C)
        target : (B, J, C)
        vis:     (B, J, 1)
    '''
    if type(source) == torch.Tensor:
        source = source.numpy()
        target = target.numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(source, axis=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    X0 /= normX
    Y0 /= normY
    
    H = np.matmul(X0.transpose(0, 2, 1), Y0) 
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    return torch.from_numpy(R)
    


def align_target_numpy(cfg, source, target):
    '''
    Args:
        source : (B, T, J, C, N)
        target : (B, T, J, C, N)
    '''
    B, T, J, C, N = source.shape
    source = source.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    target = target.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    if type(source) == torch.Tensor:
        
        # source = source.cpu().detach().numpy()
        # target = target.cpu().detach().numpy()
        source = source.numpy()
        target = target.numpy()

    muX = np.mean(target, axis=1, keepdims=True) #(B*T*N, 1, C)
    muY = np.mean(source, axis=1, keepdims=True) #(B*T*N, 1, C)
    X0 = target - muX #(B*T*N, J, C)
    Y0 = source - muY #(B*T*N, J, C)

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    X0 /= normX #(B*T*N, J, C)
    Y0 /= normY #(B*T*N, J, C)
    
    H = np.matmul(X0.transpose(0, 2, 1), Y0) #(B*T*N, C, C)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    
    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    if cfg.TEST.TRJ_ALIGN_R:
        source = np.matmul(source, R)
    if cfg.TEST.TRJ_ALIGN_S:
        source = a * source
    if cfg.TEST.TRJ_ALIGN_T:
        source = source + t
    
    source = torch.from_numpy(source)
    source = source.view(B, T, N, J, C).permute(0, 1, 3, 4, 2)  #B, T, J, C, N

    return source


def align_target_torch(cfg, source, target):
    '''
    Args:
        source : (B, T, J, C, N)
        target : (B, T, J, C, N)
    '''
    B, T, J, C, N = source.shape
    source = source.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    target = target.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C).to(source.device)

    muX = torch.mean(target, dim=1, keepdims=True) #(B*T*N, 1, C)
    muY = torch.mean(source, dim=1, keepdims=True) #(B*T*N, 1, C)
    X0 = target - muX #(B*T*N, J, C)
    Y0 = source - muY #(B*T*N, J, C)

    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    X0 /= normX #(B*T*N, J, C)
    Y0 /= normY #(B*T*N, J, C)
    
    H = torch.matmul(X0.permute(0, 2, 1), Y0) #(B*T*N, C, C)
    U, s, Vt = torch.svd(H)
    V = Vt.permute(0, 2, 1)
    R = torch.matmul(V, U.permute(0, 2, 1))
    
    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = torch.sign(torch.unsqueeze(torch.linalg.det(R), 1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= torch.flatten(sign_detR)
    R = torch.matmul(V, U.permute(0, 2, 1)) # Rotation
    
    tr = torch.unsqueeze(torch.sum(s, dim=1, keepdims=True), 2)

    a = tr * normX / normY # Scale
    t = muX - a*torch.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    if cfg.TEST.TRJ_ALIGN_R:
        source = torch.matmul(source, R)
    if cfg.TEST.TRJ_ALIGN_S:
        source = a * source
    if cfg.TEST.TRJ_ALIGN_T:
        source = source + t
    source = source.view(B, T, N, J, C).permute(0, 1, 3, 4, 2)  #B, T, J, C, N

    return source

def align_torch(source, target):
    '''
    Args:
        source : (B, J, C)
        target : (B, J, C)
        vis:     (B, J, 1)
    '''
    
    device = source.device
    B, T, J, C, N = source.shape
    source = source.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    target = target.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    assert len(source.shape) == 3

    muX = torch.mean(target, dim=1, keepdims=True)
    muY = torch.mean(source, dim=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY

    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    X0 /= normX
    Y0 /= normY
     
    H = torch.matmul(X0.permute(0, 2, 1), Y0) 
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.permute(0, 2, 1)
    R = torch.matmul(V, U.permute(0, 2, 1))
    return R
    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    
    return torch.from_numpy(R).to(device)
def test_multi_view_aug(pred, vis):
    '''
    Args:
        pred:(B, T, J, C, N) T = 1
        vis: (B, T, J, C, N) T >= 1
    '''
    B, T, J, C, N = pred.shape
    
    pad = T // 2
    if vis is not None:
        vis = vis[:,pad:pad+1] #(B, 1, J, 1, N)
    else:
        vis = torch.ones(B, 1, J, 1, N)
    att = vis.view(B, T, J, 1, 1, N).repeat(1, 1, 1, 1, N, 1)

    if N == 1:
        return pred
    else:
        final_out = torch.zeros(B*T, J, C, N, N).float()
        pred = pred.view(B*T, J, C, N)
        for view_id in range(N):
            final_out[:,:,:,view_id, view_id] = pred[:,:,:,view_id]
            
        for view_list in itertools.combinations(list(range(N)), 2):
            view_1_id = view_list[0]
            view_2_id = view_list[1]

            R = align_numpy(source=pred[:,:,:,view_2_id], target=pred[:,:,:,view_1_id])

            final_out[:,:,:,view_1_id, view_2_id] = torch.matmul(pred[:,:,:,view_2_id], R)
            final_out[:,:,:,view_2_id, view_1_id] = torch.matmul(pred[:,:,:,view_1_id], R.permute(0, 2, 1)) 

        att = F.softmax(att, dim=-1).float() #(B, T, J, C, N, N)  
        final_out = final_out.view(B, T, J, C, N, N) * att
        final_out = torch.sum(final_out, dim = -1)
        
        return final_out

        
def mpjpe(predicted, target):
    assert predicted.shape == target.shape
    l=torch.norm(predicted - target, dim=len(target.shape)-1)
    return torch.mean(l)
def pjpe(predicted, target):
    assert predicted.shape == target.shape
    l = torch.norm(predicted - target, dim=len(target.shape) - 1)
    return l

def mpjpe_per_view(predicted, target):
    assert predicted.shape == target.shape
    l=torch.norm(predicted - target, dim=len(target.shape)-1)
    return torch.mean(l, dim=(0, 1, 2))

def hubert_mpjpe(predicted, target):
    assert predicted.shape == target.shape
    return torch.nn.HuberLoss()(predicted, target)

def mpjpe_adaptive(predicted, target, adaptive):
    assert predicted.shape == target.shape
    l = torch.unsqueeze(torch.flatten(torch.norm(predicted - target, dim=len(target.shape)-1)).contiguous(), 1)
    l=adaptive.lossfun(l)
    return torch.mean(l)

def mpjpe_inverted_clip(predicted, target):
    assert predicted.shape == target.shape
    #print(predicted[0])
    #print(target[0])
    dist = torch.unsqueeze(torch.norm(predicted - target, dim=len(target.shape)-1, p=0.5), -1)
    dist = torch.where(dist>0.5*target, dist, 0)
    return torch.mean(dist)

def mpjpe_kth_best_views(predicted, target, n_best_views=4):
    assert predicted.shape == target.shape
    assert n_best_views>=2
    dist = torch.norm(predicted - target, dim=len(target.shape) - 1)
    res=torch.kthvalue(dist, 1,  dim=1)[0]+torch.kthvalue(dist, 2,  dim=1)[0]
    for k in range(3, n_best_views+1):
        res = res + torch.kthvalue(dist, k,  dim=1)[0]
    res = res/n_best_views

    return torch.mean(res)

def mpjpe_kth_best_confidence(predicted, target, confidences=None):
    """

    :param predicted: (B, T, N_views, J, C), ex (720, 1, 4, 17, 2)
    :param target: (B, T, N_views, J, C), ex : (720, 1, 4, 17, 2)
    :param confidences: (B, T, J, 1, N_views), ex : (360, 1, 17, 1, 4)
    :return:
    """
    assert predicted.shape == target.shape
    confidences = confidences.permute(0, 1, 4, 2, 3)
    max_conf_among_views = torch.max(confidences, dim=2, keepdim=True)[0].repeat(1, 1, 4, 1, 1)
    confidences = confidences/max_conf_among_views
    dist = torch.norm(predicted - target, dim=len(target.shape) - 1, keepdim=True)
    dist = torch.where(confidences.to(dist.device)==1, dist, 0)
    dist = torch.mean(dist, dim=(0, 1, 3, 4))
    dist = torch.sum(dist)
    return dist

def mpjpe_threhold_confidence(predicted, target, confidences=None, threshold=0.6):
    """

    :param predicted: (B, T, N_views, J, C), ex (720, 1, 4, 17, 2)
    :param target: (B, T, N_views, J, C), ex : (720, 1, 4, 17, 2)
    :param confidences: (B, T, J, 1, N_views), ex : (360, 1, 17, 1, 4)
    :return:
    """
    assert predicted.shape == target.shape
    confidences = confidences.permute(0, 1, 4, 2, 3).contiguous()
    thresholded_confidences = torch.where(confidences>=threshold, 1, 0).to(predicted.device).detach()
    n_views = confidences.shape[2]
    print(thresholded_confidences)
    remaining_views = torch.sum(thresholded_confidences, dim=2, keepdim=True).repeat(1, 1, n_views, 1, 1).to(predicted.device).detach()
    print(remaining_views)
    raw_dist = torch.norm(predicted - target, dim=len(target.shape) - 1, keepdim=True)
    dist = raw_dist*thresholded_confidences
    remaining_views_ = (torch.squeeze(remaining_views[:, :, 0, :, :]).detach().cpu()).numpy()
    print(remaining_views_.shape)
    plt.subplots_adjust(hspace=0.8)
    for j in range(17):
        ax = plt.subplot(4, 5, j + 1)
        ax.hist((remaining_views_[:, j]).flatten())
    plt.show()

    input()
    print(dist)
    print(torch.max(remaining_views))
    print(torch.min(remaining_views))
    dist=torch.where(remaining_views>0, torch.div(dist, remaining_views+0.01), 0)
    # print(dist)
    input()
    dist = torch.mean(dist, dim=(0, 1, 3, 4))
    dist = torch.sum(dist)
    return dist

def mpjpe_confidence_deviation_to_max(predicted, target, confidences=None, threshold=0.6):
    """

    :param predicted: (B, T, N_views, J, C), ex (720, 1, 4, 17, 2)
    :param target: (B, T, N_views, J, C), ex : (720, 1, 4, 17, 2)
    :param confidences: (B, T, J, 1, N_views), ex : (360, 1, 17, 1, 4)
    :return:
    """
    assert predicted.shape == target.shape
    confidences = confidences.permute(0, 1, 4, 2, 3).contiguous()
    max_confidences_among_views = torch.max(confidences, dim=2, keepdim=True)[0].repeat(1, 1, 4, 1, 1)
    deviations_to_max_confidence = torch.where(max_confidences_among_views>0, (max_confidences_among_views-confidences)/(max_confidences_among_views+0.001), 0)

    thresholded_confidences = torch.where(deviations_to_max_confidence<=threshold, 1, 0).to(predicted.device).detach()
    n_views = confidences.shape[2]
    remaining_views = torch.sum(thresholded_confidences, dim=2, keepdim=True).repeat(1, 1, n_views, 1, 1).to(predicted.device).detach()
    raw_dist = torch.norm(predicted - target, dim=len(target.shape) - 1, keepdim=True)
    dist = raw_dist*thresholded_confidences

    ##### "for loss validation" ######
    # remaining_views_ = (torch.squeeze(remaining_views[:, :, 0, :, :]).detach().cpu()).numpy()
    # print(remaining_views_.shape)
    # plt.subplots_adjust(hspace=0.8)
    # for j in range(17):
    #     ax = plt.subplot(4, 5, j + 1)
    #     ax.hist((remaining_views_[:, j]).flatten())
    # plt.show()
    #
    # input()
    # print(dist)
    # print(torch.max(remaining_views))
    # print(torch.min(remaining_views))
    ##### end of "for loss validation" ######

    dist=torch.where(remaining_views>0, torch.div(dist, remaining_views), 0)
    # print(dist)
    dist = torch.mean(dist, dim=(0, 1, 3, 4))
    dist = torch.sum(dist)
    return dist

def weighted_mpjpe(predicted, target, confidences=None, softmax=True, temperature=1/5):
    """

    :param predicted: (B, T, N_views, J, C), ex (720, 1, 4, 17, 2)
    :param target: (B, T, N_views, J, C), ex : (720, 1, 4, 17, 2)
    :param confidences: (B, T, J, 1, N_views), ex : (360, 1, 17, 1, 4)
    :return:
    """
    assert predicted.shape == target.shape
    confidences = confidences.permute(0, 1, 4, 2, 3)
    m = torch.nn.Softmax(dim=2)
    if softmax:
        soft_confidences = m(confidences/temperature)
    else:
        soft_confidences = confidences
    import matplotlib.pyplot as plt
    # plt.figure("confidences")
    # plt.bar(np.arange(4), confidences[0, 0,:, 10, 0])
    # plt.figure("soft confidences")
    # plt.bar(np.arange(4), soft_confidences[0, 0, :, 10, 0])
    # plt.show()
    # soft_confidences = soft_confidences.repeat(1, 1, 1, 1, 2)
    # print(soft_confidences.shape)
    # input()
    #max_conf_among_views = torch.max(confidences, dim=2, keepdim=True)[0].repeat(1, 1, 4, 1, 1)
    #confidences = confidences/max_conf_among_views
    dist = torch.norm(predicted - target, dim=len(target.shape) - 1, keepdim=True)*soft_confidences.to(predicted.device).detach()
    #dist = torch.where(confidences.to(dist.device)==1, dist, 0)
    dist = torch.mean(dist, dim=(0, 1, 3, 4))
    dist = torch.sum(dist)
    return dist

def get_batch_bones_lens(out, bones):
    bones_lens = torch.norm(out[:, :, :, bones[:, 0]] - out[:, :, :, bones[:, 1]], dim=-1)
    return bones_lens

def get_bone_prior_loss(out, bones, mean_bones_len_prior, std_bones_len_prior):
    bones_lens = get_batch_bones_lens(out, bones)
    n_bones = bones.shape[0]
    std_bones_len_prior = std_bones_len_prior.view(1, 1, 1, n_bones)
    std_bones_len_prior = std_bones_len_prior.repeat(bones_lens.shape[:3]+(1,))
    mean_bones_len_prior = mean_bones_len_prior.view(1, 1, 1, n_bones)
    mean_bones_len_prior = mean_bones_len_prior.repeat(bones_lens.shape[:3] + (1,))
    print("bones : ", bones)
    print("std_bones_len_prior[0] : ", std_bones_len_prior[0])
    print("mean_bones_len_prior[0] : ", mean_bones_len_prior[0])
    print("bones_lens[0] : ", bones_lens[0])
    mse_bone_prior = torch.square(bones_lens-mean_bones_len_prior)
    mse_bone_prior = torch.minimum(mse_bone_prior, torch.ones_like(mse_bone_prior ))
    bone_prior_loss = -(1/(2*std_bones_len_prior))*(mse_bone_prior)#*#-(3/2)*torch.log(2*math.pi)-3*torch.log(std_bones_len_prior)
    #bone_prior_loss = torch.minimum(-1*bone_prior_loss, torch.ones_like(bone_prior_loss))
    #max_val = 1/(2*torch.min(std_bones_len_prior))
    return -1*torch.mean(bone_prior_loss, dim=(0,1,3))#/max_val

# def get_bone_prior_loss(out, bones, mean_bones_len_prior, std_bones_len_prior):
#     bones_lens = get_batch_bones_lens(out, bones)
#     n_bones = bones.shape[0]
#     min_std_bone = torch.min(std_bones_len_prior)
#     max_std_bone = torch.max(std_bones_len_prior)
#     std_bones_len_prior = (std_bones_len_prior-min_std_bone)/(max_std_bone-min_std_bone)+1
#     std_bones_len_prior = std_bones_len_prior.view(1, 1, 1, n_bones)
#     std_bones_len_prior = std_bones_len_prior.repeat(bones_lens.shape[:3]+(1,))
#     mean_bones_len_prior = mean_bones_len_prior.view(1, 1, 1, n_bones)
#     mean_bones_len_prior = mean_bones_len_prior.repeat(bones_lens.shape[:3] + (1,))
#
#     # std_bones_len_prior = std_bones_len_prior/mean_bones_len_prior
#     # mean_bones_len_prior = mean_bones_len_prior/mean_bones_len_prior
#     # bones_lens = bones_lens/mean_bones_len_prior
#
#     mse_bone_prior = torch.square(bones_lens-mean_bones_len_prior)
#     mse_bone_prior = torch.minimum(mse_bone_prior, torch.ones_like(mse_bone_prior ))
#     bone_prior_loss = -(1/(2*std_bones_len_prior))*(mse_bone_prior)#*#-(3/2)*torch.log(2*math.pi)-3*torch.log(std_bones_len_prior)
#     #min_loss = torch.min(bone_prior_loss, dim=(0, 1, 2))
#
#     #bone_prior_loss = torch.minimum(-1*bone_prior_loss, torch.ones_like(bone_prior_loss))
#     #max_val = 1/(2*torch.min(std_bones_len_prior))
#     return -1*torch.mean(bone_prior_loss)#/max_val

def get_batch_bones_directions(out, bones, batch_subjects=None, cfg=None, eval=False):
    direction_vectors = out[:, :, :, bones[:, 0]] - out[:, :, :, bones[:, 1]]
    bones_lens = torch.unsqueeze(torch.norm(direction_vectors, dim=-1), -1).repeat(1, 1, 1, 1, 2)
    direction_vectors = (0.0001+direction_vectors)/(0.0001+bones_lens)
    return direction_vectors

def get_batch_body_directions(out):
    extreme_joints = [[8, 16], [8, 13], [8, 3], [8, 6]]
    extreme_joints = torch.from_numpy(np.array(extreme_joints)).to(out.device)
    direction_vectors = out[:, :, :, extreme_joints[:, 0]] - out[:, :, :, extreme_joints[:, 1]]
    bones_lens = torch.unsqueeze(torch.norm(direction_vectors, dim=-1), -1).repeat(1, 1, 1, 1, 2)
    direction_vectors = (0.0001+direction_vectors)/(0.0001+bones_lens)
    return direction_vectors


# def symetry_loss(out, bones, l_bones, r_bones):
#     l_bones_lens = torch.norm(out[:, :, :, bones[l_bones, 0]] - out[:, :, :, bones[l_bones, 1]], dim=-1)
#     r_bones_lens = torch.norm(out[:, :, :, bones[r_bones, 0]] - out[:, :, :, bones[r_bones, 1]], dim=-1)
#     #mean_bones_lens = (torch.mean(l_bones_lens, dim=(0, 1, 2))+torch.mean(r_bones_lens, dim=(0, 1, 2)))/2
#     #mean_bones_lens = mean_bones_lens.view(1, 1, 1, mean_bones_lens.shape[0])
#     #mean_bones_lens = mean_bones_lens.repeat(*l_bones_lens.shape[:3]+(1,))
#     #print(mean_bones_lens.shape)
#     #l_bones_lens = torch.minimum(l_bones_lens, torch.ones_like(l_bones_lens))
#     #r_bones_lens = torch.minimum(r_bones_lens, torch.ones_like(r_bones_lens))
#     #return torch.mean((torch.norm(torch.unsqueeze(l_bones_lens, -1)-torch.unsqueeze(r_bones_lens, -1), dim=-1))/(mean_bones_lens))
#     #return torch.mean(torch.norm(torch.unsqueeze(l_bones_lens, -1)-torch.unsqueeze(r_bones_lens, -1), dim=-1))
#     return torch.mean(torch.norm(l_bones_lens - r_bones_lens, dim=-1))
#     #return torch.mean(torch.square(torch.unsqueeze(l_bones_lens, -1)-torch.unsqueeze(r_bones_lens, -1)))

def symetry_loss(out, bones, l_bones, r_bones):
    l_bones_lens = torch.norm(out[:, :, :, bones[l_bones, 0]] - out[:, :, :, bones[l_bones, 1]], dim=-1)
    r_bones_lens = torch.norm(out[:, :, :, bones[r_bones, 0]] - out[:, :, :, bones[r_bones, 1]], dim=-1)
    return torch.mean(torch.norm(torch.unsqueeze(l_bones_lens, -1)-torch.unsqueeze(r_bones_lens, -1), dim=-1))

# def bone_losses(out, bones, batch_subjects=None, cfg=None, eval=False, get_per_view_values=False):
#     """
#     :param out: (Batch, T, N, J, C)
#     :param bones:
#     :param batch_subjects:
#     :param cfg:
#     :return:
#     """
#     if get_per_view_values:
#         dims_compute_stats = (0, 1)
#     else:
#         dims_compute_stats = (0, 1, 2)
#
#     if batch_subjects==None:
#         bones_lens = torch.norm(out[:,:,:, bones[:, 0]] - out[:,:,:, bones[:, 1]], dim=-1)
#         return torch.mean(bones_lens, dim=dims_compute_stats), torch.std(bones_lens, dim=dims_compute_stats)
#     else:
#         res_std = []
#         res_mean = []
#         if eval:
#             subjects_ = cfg.H36M_DATA.SUBJECTS_TEST
#         else:
#             subjects_ = cfg.H36M_DATA.SUBJECTS_TRAIN
#         for op in subjects_:
#             # print(np.array(batch_subjects)[:, 0])
#             # print(op)
#             # input()
#             op_out = out[np.where((np.array(batch_subjects)[:, 0])==op)]
#             # print(op_out.shape)
#             if op_out.shape[0]>0:
#                 op_bones_lens = torch.norm(op_out[:, :, :, bones[:, 0]] - op_out[:, :, :, bones[:, 1]], dim=len(op_out.shape) - 1)
#                 op_bones_lens_maxed = torch.minimum(op_bones_lens, torch.ones_like(op_bones_lens))
#                 res_std.append(torch.unsqueeze(torch.std(op_bones_lens_maxed, dim=(0, 1, 2)), dim=0))
#                 res_mean.append(torch.unsqueeze(torch.mean(op_bones_lens, dim=(0, 1, 2)), dim=0))
#                 # print(res_std[-1].shape)
#             # input()
#         mean_bone_len = torch.cat(res_mean, dim=0)
#         std_bone_len = torch.cat(res_std, dim=0)
#         #std_bone_len = (std_bone_len)/(mean_bone_len)
#         # print(mean_bone_len.shape)
#         # print(std_bone_len.shape)
#         # input()
#         return mean_bone_len, std_bone_len

def bone_losses(out, bones, subjects_, batch_subjects=None, cfg=None, eval=False, get_per_view_values=False):
    """

    :param out: (Batch, T, N, J, C)
    :param bones:
    :param batch_subjects:
    :param cfg:
    :return:
    """
    if get_per_view_values:
        dims_compute_stats = (0, 1)
    else:
        dims_compute_stats = (0, 1, 2)
    if batch_subjects==None:
        bones_lens = torch.norm(out[:,:,:, bones[:, 0]] - out[:,:,:, bones[:, 1]], dim=-1)
        return torch.mean(bones_lens, dim=dims_compute_stats), torch.std(bones_lens, dim=dims_compute_stats)
    else:

        res_std = []
        res_mean = []
        """if eval:
            subjects_ = cfg.H36M_DATA.SUBJECTS_TEST
        else:
            subjects_ = cfg.H36M_DATA.SUBJECTS_TRAIN"""


        for op in subjects_:
            # print(np.array(batch_subjects)[:, 0])
            # print(op)
            # input()
            op_out = out[np.where((np.array(batch_subjects)[:, 0])==op)]
            if op_out.shape[0]>0:
                op_bones_lens = torch.norm(op_out[:, :, :, bones[:, 0]] - op_out[:, :, :, bones[:, 1]], dim=len(op_out.shape) - 1)
                op_bones_lens_maxed = torch.minimum(op_bones_lens, torch.ones_like(op_bones_lens))
                res_std.append(torch.unsqueeze(torch.std(op_bones_lens_maxed, dim=dims_compute_stats), dim=0))
                res_mean.append(torch.unsqueeze(torch.mean(op_bones_lens, dim=dims_compute_stats), dim=0))
                # print(res_std[-1].shape)

            # input()
        mean_bone_len = torch.cat(res_mean, dim=0)
        std_bone_len = torch.cat(res_std, dim=0)
        # print(mean_bone_len.shape)
        # print(std_bone_len.shape)
        # input()
        return mean_bone_len, std_bone_len

def params_regularization(params, subjects_, batch_subjects=None, cfg=None, eval=False):
    """
    :param batch_subjects:
    :param cfg:
    :return:
    """

    params_std = []
    #res_mean = []
    """if eval:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TEST
    else:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TRAIN"""
    for op in subjects_:
        # print(np.array(batch_subjects)[:, 0])
        # print(op)
        # input()
        op_params = params[np.where((np.array(batch_subjects)[:, 0])==op)]
        if op_params.shape[0]>0:
            op_params_std = torch.std(op_params, dim=(0, 1))
            params_std.append(torch.unsqueeze(torch.mean(op_params_std), 0))
    return torch.mean(torch.cat(params_std))

def bone_len_loss(bone_priors, out, bones, subjects_, batch_subjects=None, cfg=None, eval=False, std_bones_len_prior=None):
    n_bones = bones.shape[0]
    std_bones_len_prior = std_bones_len_prior.view(1, 1, 1, n_bones)

    loss = []
    """if eval:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TEST
    else:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TRAIN"""
    for op in subjects_:
        # print(np.array(batch_subjects)[:, 0])
        # print(op)
        # input()
        op_out = out[np.where((np.array(batch_subjects)[:, 0]) == op)]
        # print(op_out.shape)
        #print(op)
        if op_out.shape[0] > 0:
            op_bones_lens = torch.norm(op_out[:, :, :, bones[:, 0]] - op_out[:, :, :, bones[:, 1]], dim=len(op_out.shape) - 1)
            prior_len = torch.reshape(bone_priors[op].to(op_bones_lens.device), (1, 1, 1, bones.shape[0])).repeat(op_out.shape[:-2]+(1,))
            #print(prior_len[0, 0, 0])
            #print(op_bones_lens[0, 0, 0])
            #prior_len = torch.unsqueeze(prior_len, -1)
            #op_bones_lens = torch.unsqueeze(op_bones_lens, -1)
            #loss.append(torch.unsqueeze(mpjpe(prior_len.to(out.device), op_bones_lens), 0))
            mse=torch.norm(torch.unsqueeze(prior_len.to(out.device), -1) - torch.unsqueeze(op_bones_lens, -1), dim=-1)
            #mse = torch.square(prior_len.to(out.device) - op_bones_lens)
            mse = torch.minimum(mse, torch.ones_like(mse)*0.5)
            std_bones_len_prior_ = std_bones_len_prior.repeat(mse.shape[:3] + (1,))
            #loss.append((1 / (2 * std_bones_len_prior_))*mse)
            loss.append(mse)
    return torch.mean(torch.cat(loss))

def bone_len_mae(bone_priors, out, bones, subjects_, batch_subjects=None, avg_ov_frames=True, remove_fails_from_stats=False):
    n_bones = bones.shape[0]
    loss = []
    """if eval:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TEST
    else:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TRAIN"""
    for op in subjects_:
        # print(np.array(batch_subjects)[:, 0])
        # print(op)
        # input()
        op_out = out[np.where((np.array(batch_subjects)[:, 0]) == op)]
        # print(op_out.shape)
        #print(op)
        if op_out.shape[0] > 0:
            op_bones_lens = torch.norm(op_out[:, :, :, bones[:, 0]] - op_out[:, :, :, bones[:, 1]], dim=len(op_out.shape) - 1)
            prior_len = torch.reshape(bone_priors[op].to(op_bones_lens.device), (1, 1, 1, bones.shape[0])).repeat(op_out.shape[:-2]+(1,))
            mae=torch.abs(prior_len.to(out.device) - op_bones_lens)
            #get values where mae>prior_len/2
            # print(prior_len)
            # print(mae)
            # input()
            if remove_fails_from_stats:
                mae = torch.where(mae > prior_len/2, torch.zeros_like(mae)*float('nan'), mae)
            loss.append(mae)
    if avg_ov_frames:
        return torch.mean(torch.cat(loss), dim=1)
    else:
        return torch.cat(loss)

def get_bones_lens(bone_priors_mean, bone_priors_std, out, bones, subjects_, batch_subjects=None, cfg=None, eval=False):
    """if eval:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TEST
    else:
        subjects_ = cfg.H36M_DATA.SUBJECTS_TRAIN"""
    for op in subjects_:
        # print(np.array(batch_subjects)[:, 0])
        # print(op)
        # input()
        op_out = out[np.where((np.array(batch_subjects)[:, 0]) == op)]
        # print(op_out.shape)

        if op_out.shape[0] > 0:
            op_bones_lens = torch.norm(op_out[:, :, :, bones[:, 0]] - op_out[:, :, :, bones[:, 1]], dim=len(op_out.shape) - 1)
            bone_priors_mean[op].append(torch.unsqueeze(torch.mean(op_bones_lens, dim=(0, 1, 2)), dim=0).detach().cpu().numpy())
            bone_priors_std[op].append(torch.unsqueeze(torch.std(op_bones_lens, dim=(0, 1, 2)), dim=0).detach().cpu().numpy())

    return bone_priors_mean, bone_priors_std


def p_mpjpe(cfg, predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    if type(predicted) == torch.Tensor:
        predicted = predicted.numpy()
    if type(target) == torch.Tensor:
        target = target.numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    if cfg.TEST.METRIC_ALIGN_R:
        predicted = np.matmul(predicted, R)
    if cfg.TEST.METRIC_ALIGN_S:
        predicted = a * predicted
    if cfg.TEST.METRIC_ALIGN_T:
        predicted = predicted + t
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1))

def p_mpjpe_per_view(cfg, predicted, target, n_views=4):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    if type(predicted) == torch.Tensor:
        predicted = predicted.numpy()
    if type(target) == torch.Tensor:
        target = target.numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    if cfg.TEST.METRIC_ALIGN_R:
        predicted = np.matmul(predicted, R)
    if cfg.TEST.METRIC_ALIGN_S:
        predicted = a * predicted
    if cfg.TEST.METRIC_ALIGN_T:
        predicted = predicted + t
    l=np.linalg.norm(predicted - target, axis=len(target.shape) - 1)
    l=l.reshape((-1, n_views, l.shape[-1]))
    return np.mean(l, axis=(0, 2))

def p_mpjpe_torch(cfg, predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = torch.mean(target, dim=1, keepdims=True)
    muY = torch.mean(predicted, dim=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

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
    
    # Perform rigid transformation on the input
    if cfg.TEST.METRIC_ALIGN_R:
        predicted = torch.matmul(predicted, R_2)
    if cfg.TEST.METRIC_ALIGN_S:
        predicted = a * predicted
    if cfg.TEST.METRIC_ALIGN_T:
        predicted = predicted + t


    return torch.mean(torch.linalg.norm(predicted - target, dim=len(target.shape)-1))

def mv_mpjpe(pred, gt, mask):
    """
    pred:(B, T, N, J, C)
    gt: (B, T, N, J, C)
    mask: (B, N, N)
    """

    B, T, N, J, C = pred.shape
    
    loss = 0
    pad = T // 2
    
    num = 0
    for b in range(mask.shape[0]):
        for view_pair in itertools.combinations(list(range(mask.shape[-1])), 2):
            view_1_id = view_pair[0]
            view_2_id = view_pair[1]
            m_1 = mask[b, view_1_id]
            m_2 = mask[b, view_2_id]
            if torch.equal(m_1, m_2):
                R = align_numpy(source=gt[b:b+1, 0, view_2_id].cpu(), target=gt[b:b+1, 0, view_1_id].cpu())
                tmp = torch.einsum('btjc,bck->btjk', pred[b:b+1,:,view_2_id], R.to(pred.device))

                loss = loss + mpjpe(tmp, pred[b:b+1,:,view_1_id])
                num += 1

    return loss / (num + 1e-9)

def get_rotation(target):
    #B, T, J, C, N

    device = target.device
    target = target.permute(0, 1, 4, 2, 3) #(B, T, N, J, C)
    B, T, N, J, C = target.shape
    predicted = target.view(B, T, 1, N, J, C).repeat(1, 1, N, 1, 1, 1).view(-1, J, C)
    target = target.view(B, T,N,1, J, C).repeat(1, 1, 1, N, 1, 1).view(-1, J, C)
    
    if type(predicted) == torch.Tensor:
        predicted = predicted.detach().cpu().numpy()
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
        
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    
    R = torch.from_numpy(R).float().to(device)
    R = R.view(B, T, N, N, 3, 3)
    R = R.permute(0, 5, 4, 1, 2, 3) #(B,  3, 3,T, N, N,)

    return R


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted**2, dim=-1, keepdim=True), dim=-2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=-1, keepdim=True), dim=-2, keepdim=True)
    scale = norm_target / norm_predicted
    
    return mpjpe(scale * predicted, target)



def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))
