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

    # def get_xyz_proj(self, uv1):
    #     sensor_XY = uv1[:,:,:,:2] * self.arg.proj_pitch # B, m, wvl, uv1, # px
    #     sensor_Z = torch.zeros_like(sensor_XY[:,:,:,0])
    #     sensor_Z[:] = self.focal_length_proj()
        
    #     xyz_proj = torch.stack((sensor_XY[:,:,:,0], sensor_XY[:,:,:,1], sensor_Z), dim = 3)

    #     return xyz_proj     
    
    # def make_uv1(self, sensor_U_distorted, sensor_V_distorted, zero_uv1):
    #     ones_uv = torch.stack((sensor_U_distorted, sensor_V_distorted), dim = 3)
        
    #     zero_uv = zero_uv1[:,:2].unsqueeze(dim = 1).unsqueeze(dim = 1) # B, uv, # px
    #     zero_uv = zero_uv.repeat(1,1,25,1,1)

    #     uv = torch.cat((ones_uv[:,0].unsqueeze(dim = 1), zero_uv[:,0].unsqueeze(dim = 1), ones_uv[:,1].unsqueeze(dim = 1)), dim = 1)
    #     ones = torch.ones_like(uv[:,:,:,0])
        
    #     uv1 = torch.stack((uv[:,:,:,0], uv[:,:,:,1], ones), dim = 3)

    #     return uv1
    
    # def projection(self, X, Y, Z):
    #     XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 1).to(device=self.device)
    #     XYZ1_proj = torch.linalg.inv(self.extrinsic_proj_real())@XYZ1       

    #     suv = self.intrinsic_proj_real().to(self.device) @ XYZ1_proj[:,:3]
    #     zero_uv1 = suv/ suv[...,2,:].unsqueeze(dim = 1)
        
    #     return zero_uv1
    
    
    def cubic_interpolation(self, x_new, x_points, y_points, n):
        tck = interpolate.splrep(x_points, y_points, k=n)   # Estimate the polynomial of nth degree by using x_points and y_points
        y_new = interpolate.splev(x_new, tck)
        return y_new

    def get_PEF(self):
        # get original PEF (with out optimization)
        PEF = np.load(os.path.join(self.response_function_dir, 'CRF_proj.npy'))
        map_scale = interp1d([PEF.min(), PEF.max()], [0.,1.])
        PEF = torch.tensor(map_scale(PEF).astype(np.float32))
        PEF = PEF[3:27] 
        
        PEF_R = self.cubic_interpolation(self.new_wvls, self.wvls, PEF[:,0], 4)
        PEF_G = self.cubic_interpolation(self.new_wvls, self.wvls, PEF[:,1], 4)
        PEF_B = self.cubic_interpolation(self.new_wvls, self.wvls, PEF[:,2], 4)

        PEF_intp = np.stack((PEF_R, PEF_G, PEF_B))

        optimized_pef = PEF_intp.T * self.opt_param_final[:,:3]
        
        return optimized_pef

    def get_dg_intensity(self):
        dg_intensity = np.load(os.path.join(self.response_function_dir, 'dg_efficiency.npy'))[1:]
        
        # interpolated CRF for 5nm
        dg_efficiency_R = self.cubic_interpolation(self.new_wvls, self.wvls, dg_intensity[:,0], 4)
        dg_efficiency_G = self.cubic_interpolation(self.new_wvls, self.wvls, dg_intensity[:,1], 4)
        dg_efficiency_B = self.cubic_interpolation(self.new_wvls, self.wvls, dg_intensity[:,2], 4)

        dg_intensity_intp = np.stack((dg_efficiency_R, dg_efficiency_G, dg_efficiency_B))
        
        dg_intensity_intp[2] = dg_intensity_intp[2] * self.opt_param_final[:,-1]
        dg_intensity_intp[0] = dg_intensity_intp[0] * self.opt_param_final[:,-2]

        optimized_dg = dg_intensity_intp

        return optimized_dg