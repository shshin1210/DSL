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
        
        # path 
        self.crf_dir = arg.projector_response
        self.dg_intensity_dir = arg.dg_intensity_dir
        
        # projector intrinsic
        self.focal_length_proj = arg.focal_length_proj*1e-3

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
                                                
    #### Extrinsic matrix ( cam = E @ proj // E : projector coord to world coord )
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
    
    #### Extrinsic matrix ( proj = E @ dg // E : dg coord to projector coord)
    def extrinsic_diff(self):
        # rotation, translation matrix
        extrinsic_diff = torch.zeros((4,4), device= self.device)
        
        # extrinsic_diff[:3,:3] = torch.tensor([[9.9999821e-01 ,1.8362569e-03,-4.4727555e-04],
        #                                          [-1.8364087e-03 ,9.9999827e-01 ,  -3.3851352e-04],
        #                                         [ 4.4665168e-04,3.3933623e-04 ,9.9999982e-01]])
        extrinsic_diff[:3,:3] = torch.tensor([[ 9.9999732e-01,  2.2625325e-03, -5.3116900e-04],
                                                [-2.2627593e-03,  9.9999732e-01, -4.2562018e-04],
                                                [ 5.3020113e-04,  4.2682525e-04 , 9.9999976e-01]])
        
        # translate 
        # t_mtrx = torch.tensor([[0.],[0.],[-5.8818672e-02]])
        t_mtrx = torch.tensor([[0.],[0.],[-5.8024708e-02]])
            
        extrinsic_diff[:3,3:4] = t_mtrx
        extrinsic_diff[3,3] = 1
        extrinsic_diff = torch.linalg.inv(extrinsic_diff)
        
        return extrinsic_diff

    def XYZ_to_dg(self, X,Y,Z):
        """
            Input : world coordinate X,Y,Z 3d points
            output : dg coordinate X,Y,Z 3d points
        """
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 1).to(device=self.device)
        
        # world coord XYZ1 to projector coord
        XYZ1_proj = torch.linalg.inv(self.extrinsic_proj_real())@XYZ1            
        
        # proj coord XYZ1 to dg coord XYZ1
        XYZ1_dg = torch.linalg.inv(self.extrinsic_diff())@XYZ1_proj
        
        return XYZ1_dg[:,:3]

    def intersections_dg(self, optical_center_virtual, XYZ_dg):
        optical_center_virtual = optical_center_virtual.unsqueeze(dim = 0).unsqueeze(dim = 4)
        XYZ_dg = XYZ_dg.unsqueeze(dim = 1).unsqueeze(dim = 1)
        
        dir_vec = XYZ_dg - optical_center_virtual
        norm = dir_vec.norm(dim = 3)
        dir_vec_unit = dir_vec/norm.unsqueeze(dim = 3)
        
        t = - optical_center_virtual[...,2,:] / dir_vec_unit[...,2,:]
        intersection_points_dg = dir_vec_unit*(t.unsqueeze(dim = 3)) + optical_center_virtual

        return intersection_points_dg      
        
    def intersect_points_to_proj(self, intersection_points_dg_real1):
        intersection_points_proj_real = self.extrinsic_diff()@intersection_points_dg_real1
        
        return intersection_points_proj_real[...,:3,:]
    
    def projection(self, intersection_points_proj_real):
        dir_vec = intersection_points_proj_real
        norm = intersection_points_proj_real.norm(dim = 3)
        
        dir_vec_unit =  dir_vec/norm.unsqueeze(dim = 3)
        
        t = (self.focal_length_proj - intersection_points_proj_real[...,2,:]) / dir_vec_unit[...,2,:]
        
        xyz_proj = dir_vec_unit * t.unsqueeze(dim = 3) + intersection_points_proj_real
        
        return xyz_proj
    
    def zero_order_projection(self, X, Y, Z):
        
        # XYZ_dg -> proj coord -> center 연결 -> proj plane 만나는 점        
        # world coord XYZ1 to projector coord
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 1).to(device=self.device)
        XYZ1_proj = torch.linalg.inv(self.extrinsic_proj_real())@XYZ1       

        suv = self.intrinsic_proj_real().to(self.device) @ XYZ1_proj[:,:3]
        gt_uv1 = suv/ suv[...,2,:].unsqueeze(dim = 1)
        
        return gt_uv1
     
    def xy_to_uv(self, xyz_proj):
        suv = self.intrinsic_proj_real().to(self.device) @ xyz_proj
        uv1 = suv/ suv[...,2,:].unsqueeze(dim = 3)
        
        return uv1

    #### Get virtual projector center
    def get_virtual_center(self, P, dir, wvls_N, m_N):
        """P and dir are NxD arrays defining N lines.
        D is the dimension of the space. This function 
        returns the least squares intersection of the N
        lines from the system given by eq. 13 in 
        http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
        """
        p_list = torch.zeros(size=(m_N, wvls_N,3))

        for i in range(dir.shape[2]): # m
            for j in range(dir.shape[1]): # wvls
                dir_wvls_m = dir[:,j,i,:] # 720*720, 3
                torch_eye = torch.eye(dir_wvls_m.shape[1], device= self.device)

                # generate the array of all projectors
                projs = torch_eye - torch.unsqueeze(dir_wvls_m, dim = 2)*torch.unsqueeze(dir_wvls_m, dim = 1)  # I - n*n.T

                # see fig. 1                                

                # generate R matrix and q vector
                R = projs.sum(axis=0) 
                q = (projs @ torch.unsqueeze(P, dim = 2)).sum(axis=0) 
                    
                # solve the least squares problem for the 
                # intersection point p: Rp = q
                p = torch.linalg.lstsq(R,q,rcond=None)[0]
                p = p.squeeze()
                
                p_list[i,j,:] = p

        return p_list
    
    def get_PRF(self):
        PRF = np.load(os.path.join(self.crf_dir, 'CRF_proj.npy'))
        map_scale = interp1d([PRF.min(), PRF.max()], [0.,1.])
        PRF = torch.tensor(map_scale(PRF).astype(np.float32))    
        PRF = PRF[2:27]
        return PRF

    def get_dg_intensity(self):
        dg_intensity = np.load(os.path.join(self.dg_intensity_dir, 'intensity_dg.npy'))
        
        return dg_intensity