import torch
import numpy as np
import os
from scipy.interpolate import interp1d
from hyper_sl.utils import calibrated_params

import sys
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')

class Projector():
    def __init__(self,arg, device):
        # device
        self.device = device
        
        # arg
        self.arg = arg
        self.m_n = arg.m_num
        self.wvls_n = arg.wvl_num
        
        # path 
        self.crf_dir = arg.projector_response
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

    def get_xyz_proj(self, uv1):
        sensor_XY = uv1[:,:,:,:2] * self.arg.proj_pitch # B, m, wvl, uv1, # px
        sensor_Z = torch.zeros_like(sensor_XY[:,:,:,0])
        sensor_Z[:] = self.focal_length_proj()
        
        xyz_proj = torch.stack((sensor_XY[:,:,:,0], sensor_XY[:,:,:,1], sensor_Z), dim = 3)

        return xyz_proj     
    
    def make_uv1(self, sensor_U_distorted, sensor_V_distorted, zero_uv1):
        ones_uv = torch.stack((sensor_U_distorted, sensor_V_distorted), dim = 3)
        
        zero_uv = zero_uv1[:,:2].unsqueeze(dim = 1).unsqueeze(dim = 1) # B, uv, # px
        zero_uv = zero_uv.repeat(1,1,25,1,1)

        uv = torch.cat((ones_uv[:,0].unsqueeze(dim = 1), zero_uv[:,0].unsqueeze(dim = 1), ones_uv[:,1].unsqueeze(dim = 1)), dim = 1)
        ones = torch.ones_like(uv[:,:,:,0])
        
        uv1 = torch.stack((uv[:,:,:,0], uv[:,:,:,1], ones), dim = 3)

        return uv1
    
    def projection(self, X, Y, Z):
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 1).to(device=self.device)
        XYZ1_proj = torch.linalg.inv(self.extrinsic_proj_real())@XYZ1       

        suv = self.intrinsic_proj_real().to(self.device) @ XYZ1_proj[:,:3]
        zero_uv1 = suv/ suv[...,2,:].unsqueeze(dim = 1)
        
        return zero_uv1
    
    def get_PRF(self):
        PRF = np.load(os.path.join(self.crf_dir, 'CRF_proj.npy'))
        map_scale = interp1d([PRF.min(), PRF.max()], [0.,1.])
        PRF = torch.tensor(map_scale(PRF).astype(np.float32))    
        PRF = PRF[2:27]
        return PRF

    def get_dg_intensity(self):
        dg_intensity = np.load(os.path.join(self.dg_intensity_dir, 'intensity_dg_0503.npy'))
        
        return dg_intensity