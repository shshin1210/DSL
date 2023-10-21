import torch
import numpy as np
import os
from scipy.interpolate import interp1d
from hyper_sl.utils import calibrated_params
from scipy import interpolate

import sys
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')

class Projector():
    def __init__(self, arg, device):
        # device
        self.device = device
        
        # arg
        self.arg = arg
        self.m_n = arg.m_num
        self.wvls_n = arg.wvl_num
        self.wvls = arg.wvls
        self.new_wvls = arg.new_wvls
        
        self.opt_param_final = np.load(arg.response_function_opt_param_dir)

        # path 
        self.response_function_dir = arg.response_function_dir
        self.dg_intensity_dir = arg.dg_intensity_dir

        self.proj_int, _, self.proj_rmat, self.proj_tvec = calibrated_params.bring_params(arg.calibration_param_path,"proj")

    def intrinsic_proj_real(self):
        """
            example:
            ones = torch.ones_like(r, device=device)
            xyz_p = torch.stack((r*depth,c*depth,ones*depth), dim = 0)
            XYZ = torch.linalg.inv(proj_int)@xyz_p
        """

        intrinsic_proj_real = torch.tensor(self.proj_int).type(torch.float32)
        
        return intrinsic_proj_real
    
    # projector focal length
    def focal_length_proj(self):
        fcl = self.intrinsic_proj_real() * self.arg.proj_pitch
        
        return fcl[0][0]
                                     
    # projector coord to world coord 
    def extrinsic_proj_real(self):
        """ extrinsic_proj_real @ XYZ1 --> proj coord to world coord
        
        """
        extrinsic_proj_real = torch.zeros((4,4)).to(self.device)
        
        extrinsic_proj_real[:3,:3] = torch.tensor(self.proj_rmat).type(torch.float32)
        t_mtrx = torch.tensor(self.proj_tvec).type(torch.float32)
        
        extrinsic_proj_real[:3,3:4] = t_mtrx*1e-3
        extrinsic_proj_real[3,3] = 1
        extrinsic_proj_real = torch.linalg.inv(extrinsic_proj_real)

        return extrinsic_proj_real    

    def get_PEF(self):
        PEF = np.load(os.path.join(self.response_function_dir, 'PEF.npy'))
        
        return PEF

    def get_dg_intensity(self):
        dg_intensity = np.load(os.path.join(self.response_function_dir, 'DG.npy'))

        return dg_intensity