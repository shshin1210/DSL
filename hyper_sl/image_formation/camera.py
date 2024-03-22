import torch
import numpy as np
import os
from hyper_sl.utils import calibrated_params

class Camera():
    """
        Camera parameters
    
    """
    def __init__(self, arg):
        
        # args
        self.arg = arg
        self.m_n = arg.m_num
        self.wvls_n = arg.wvl_num
        self.wvls = arg.wvls
        self.new_wvls = arg.new_wvls
        
        # params
        self.cam_int, _ = calibrated_params.bring_params(arg.calibration_param_path,"cam")

    # intrinsic params of camera
    def intrinsic_cam(self):        
        intrinsic_cam = torch.tensor(self.cam_int).type(torch.float32).to(self.arg.device)
        
        return intrinsic_cam

    # camera response function
    def get_CRF(self):
        CRF = np.load(self.arg.crf_dir)

        return CRF
        
        