import torch
import numpy as np
import os

import sys
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')

class Projector():
    def __init__(self,arg, device):
        # device
        self.device = device
        
        # path 
        self.crf_dir = arg.projector_response
        self.dg_intensity_dir = arg.dg_intensity_dir
        
        # projector intrinsic
        self.focal_length_proj = arg.focal_length_proj*1e-3
    
    def intrinsic_proj_real(self):
        """
            example:
            ones = torch.ones_like(r, device=device)
            xyz_p = torch.stack((r*depth,c*depth,ones*depth), dim = 0)
            XYZ = torch.linalg.inv(proj_int)@xyz_p
            
        """
        # intrinsic_proj_real = torch.tensor([[1.00093845e+03 ,0.00000000e+00, 2.97948385e+02],
        #                                     [0.00000000e+00, 1.01013006e+03, 3.59948973e+02],
        #                                     [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
        
        intrinsic_proj_real = torch.tensor([[1.01413202e+03, 0.00000000e+00, 3.01185491e+02],
                                            [0.00000000e+00 ,1.01411098e+03 ,3.24341546e+02],
                                            [0.00000000e+00, 0.00000000e+00 ,1.00000000e+00]])
        
        
        
        return intrinsic_proj_real
                                                
    #### Extrinsic matrix ( cam = E @ proj // E : projector coord to world coord )
    def extrinsic_proj_real(self):
        """ extrinsic_proj_real @ XYZ1 --> proj coord to world coord
        
        """
        extrinsic_proj_real = torch.zeros((4,4)).to(self.device)
        
        # rotation
        # extrinsic_proj_real[:3,:3] = torch.tensor([[ 0.99954079, -0.00161654 , 0.03025891],
        #                                             [ 0.00558025 , 0.99131739, -0.13137238],
        #                                             [-0.02978381 , 0.1314809 ,  0.99087118]])
        
        # rotation
        extrinsic_proj_real[:3,:3] = torch.tensor([[ 0.99966018, -0.00384192 ,0.02578291],
                                                   [ 0.00631669, 0.99530308 ,-0.09660162] ,
                                                   [-0.02529067, 0.09673165 ,0.99498913]])
        
        # t_mtrx = torch.tensor([[-63.94495247],
        #                         [-12.97260334],
        #                         [-13.05130514]])
        
        t_mtrx = torch.tensor([[-62.93973922] ,[-13.57632379], [ -5.49703815]])
        
        extrinsic_proj_real[:3,3:4] = t_mtrx*1e-3
        extrinsic_proj_real[3,3] = 1
        extrinsic_proj_real = torch.linalg.inv(extrinsic_proj_real)

        return extrinsic_proj_real
    
    #### Extrinsic matrix ( proj = E @ dg // E : dg coord to projector coord)
    def extrinsic_diff(self):
        # rotation, translation matrix
        extrinsic_diff = torch.zeros((4,4), device= self.device)

        # rotation        
        # extrinsic_diff[:3,:3] = torch.tensor([[ 0.9999723, -0.00734455 , -0.00119761],
        #                                         [ 0.00734283 , 0.999972, -0.00142986],
        #                                         [ 0.00120808, 0.00142101 , 0.9999983]])       
        
        extrinsic_diff[:3,:3] = torch.tensor([[ 9.9997e-01, -6.7196e-03, -3.4573e-03],
                                                [ 6.7228e-03,  9.9998e-01,  8.8693e-04],
                                                [ 3.4512e-03, -9.1015e-04,  9.9999e-01]])
        # translate 
        # t_mtrx = torch.tensor([[0.],
        #                         [0.],
        #                         [-0.03261498]])
        
        t_mtrx = torch.tensor([[0.],[0.],[-4.2502e-02]])
    
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
        PRF = np.load(os.path.join(self.crf_dir, 'CRF_proj.npy'))* 100     
        PRF = PRF[2:27]
        return PRF

    def get_dg_intensity(self):
        dg_intensity = np.load(os.path.join(self.dg_intensity_dir, 'intensity_dg.npy'))
        
        return dg_intensity