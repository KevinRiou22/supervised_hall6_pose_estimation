#from .video_multi_view_backup import VideoMultiViewModel as MODEL
from .video_multi_view import VideoMultiViewModel as MODEL
from .fuse_views_mht import fuse_views_mht as MODEL_mht
def get_models(cfg):
    if cfg.DATA.DATASET_NAME == 'h36m':
        num_joints = cfg.H36M_DATA.NUM_JOINTS 
    elif cfg.DATA.DATASET_NAME == 'total_cap':
        num_joints = cfg.TOTALCAP_DATA.NUM_JOINTS 
    if cfg.NETWORK.TYPE == 'MHT':
        train_model = MODEL_mht(cfg, is_train=True, num_joints=num_joints)
        test_model = MODEL_mht(cfg, is_train=False, num_joints=num_joints)
    else:
        train_model = MODEL(cfg, is_train=True, num_joints=num_joints)
        test_model = MODEL(cfg, is_train=False, num_joints=num_joints)
    return train_model, test_model
