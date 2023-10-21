import torch
import numpy as np
import os
from hyper_sl.utils import calibrated_params
from scipy import interpolate

class Camera():
    def __init__(self, arg):
        
        # args
        self.arg = arg
        self.opt_param_final = np.load(arg.response_function_opt_param_dir)
        self.m_n = arg.m_num
        self.wvls_n = arg.wvl_num
        self.wvls = arg.wvls
        self.new_wvls = arg.new_wvls
        
        # path 
        self.response_function_dir = arg.response_function_dir
        
        # params
        self.cam_int, _ = calibrated_params.bring_params(arg.calibration_param_path,"cam")

    def intrinsic_cam(self):        
        intrinsic_cam = torch.tensor(self.cam_int).type(torch.float32).to(self.arg.device)
        
        return intrinsic_cam
    
    # #### Unprojection
    # def unprojection(self, depth, cam_coord):
    #     """ Unproject camera sensor coord plane to world coord

    #         input : depth
    #         return : world coordinate X,Y,Z
    #     """
        
    #     suv1 = cam_coord*depth.unsqueeze(dim = 2)
    #     XYZ = torch.linalg.inv(self.intrinsic_cam())@(suv1.permute(0,2,1))
        
    #     X, Y, Z = XYZ[:,0,:], XYZ[:,1,:], XYZ[:,2,:]
    #     return X, Y, Z

    def get_CRF(self):
        CRF = np.load(os.path.join(self.response_function_dir, 'CRF.npy'))

        return CRF
        
        