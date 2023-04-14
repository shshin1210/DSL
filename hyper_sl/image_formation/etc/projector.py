import torch
import numpy as np
import os
from scipy import interpolate
from scipy.io import loadmat
import cv2

import sys
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')
# from utils import constants as C
# from hyper_sl.image_formation.camera import Camera

class Projector():
    def __init__(self,arg, device):
        # device
        self.device = device
        
        # path 
        # self.rgb_fit_path = C.RGB_FIT_DIR
        # self.rgb_result_dir = C.RGB_RESULT_DIR
        self.crf_dir = arg.projector_response
        
        # projector intrinsic
        self.focal_length_proj = arg.focal_length_proj*1e-3
        
        # wavelengths
        # self.wvls = arg.wvls
        # self.wvls_N = len(self.wvls)
        
        # self.wvls_nm = self.wvls*1e9
        # self.wvls_nm = self.wvls_nm.long()
        
        # wavelengths for interpolation
        # self.wvls_int = torch.linspace(400*1e-9, 750*1e-9, 15)
        
        # m order
        # self.m_list = arg.m_list
        # self.m_N = len(self.m_list)
        
        # class
        # self.camera = Camera(arg)
        
        # extrinsic
        self.ext_diff = self.get_ext_diff()
    
    #### Projection
    def projection(self, X,Y,Z, vproj, focal_length_proj_virtual):
        """
            input : object's XYZ coordinate
                    extrinsic matrix of rproj to vproj (vproj)
                    focal length of virtual projector
                    
            outputs : virtual projector sensor coordinate (in virtual proj's coordinate)
        """

        # focal_length_proj_virtual : 3, 77

        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 0).to(device=self.device)

        # XYZ coords in virtual proj's coordinate                   
        XYZ_vir = torch.linalg.inv(vproj)@torch.linalg.inv(self.get_ext_proj_real())@XYZ1
        XYZ_vir = XYZ_vir[:,:,:3,:] 

        # use triangle similarity ratio
        # xy of virtual proj sensor in virtual proj coordinate
        focal_length_proj_virtual = torch.unsqueeze(focal_length_proj_virtual, dim = 2)
        focal_length_proj_virtual = torch.unsqueeze(focal_length_proj_virtual, dim = 2)
        XYZ_vir_z = torch.unsqueeze(XYZ_vir[:,:,2,:] , dim = 2)

        xy_vproj = (-focal_length_proj_virtual*XYZ_vir[:,:,:2,:]/XYZ_vir_z)

        return xy_vproj 
        
    #### Extrinsic matrix
    def extrinsic_proj_real(self):
        """ World coordinate to real proj's coordinate
        
        """
        extrinsic_proj_real = torch.zeros((4,4)).to(self.device)
        # no rotation
        extrinsic_proj_real[0,0] = 1 
        extrinsic_proj_real[1,1] = 1
        extrinsic_proj_real[2,2] = 1

        # translate + x 50e-3
        extrinsic_proj_real[0,3] = 50e-3 
        extrinsic_proj_real[3,3] = 1
        
        return extrinsic_proj_real
    
    def extrinsic_diff(self):
        # rotation, translation matrix
        extrinsic_diff = torch.zeros((4,4), device= self.device)

        # rotation
        rot = torch.tensor([1,1,1])
        extrinsic_diff[0,0] = rot[0]
        extrinsic_diff[1,1] = rot[1]
        extrinsic_diff[2,2] = rot[2]

        # translate
        trans = torch.tensor([0,0,10*1e-3])
        extrinsic_diff[0,3] = trans[0]
        extrinsic_diff[1,3] = trans[1]
        extrinsic_diff[2,3] = trans[2]
        extrinsic_diff[3,3] = 1
        
        return extrinsic_diff
    
    def vproj_to_dg(self, xy_vproj, vproj, sensor_Z_virtual):
        """ Virtual sensor in DG coord
        
            inputs : virtual projector sensor xy coordinate(virtual proj's coord)
                    extrinsic matrix of virtual proj to real proj (vproj)
                    
                    virtual projector's z coord to make xyz1

            outputs : virtual proj's sensor in DG coord
        """
        
        # xy_vproj : 3, 77 ,2, 640*640
        ones = torch.ones_like(xy_vproj[:,:,0,:])
        z_vproj = torch.zeros_like(xy_vproj[:,:,0,:])
        z_vproj[:] = -sensor_Z_virtual.mean()
        
        # 3, 77 , 3, 640*640
        xyz1_vproj = torch.stack((xy_vproj[:,:,0,:], xy_vproj[:,:,1,:],z_vproj, ones), dim = 2)

        xyz_dg = torch.linalg.inv(self.ext_diff)@vproj@xyz1_vproj
        xy_dg = xyz_dg[:,:,:2]

        return xy_dg
    
    def dg_to_rproj(self, xy_proj):
        """
        inputs : distorted virtual sensor coord = real proj plane coord (xy_proj) in dg coord

        outputs : real proj sensor plane coords in real proj coord
        """
        # xy_proj : 3, 77, 2, 640*640
        z_proj = torch.zeros_like(xy_proj[:,:,0,:])
        z_proj[:] = -self.focal_length_proj

        xyz1_proj = torch.stack((xy_proj[:,:,0,:], xy_proj[:,:,1,:], z_proj, torch.ones_like(xy_proj[:,:,0,:])), dim=2)
                    # 3, 77, 4, 640*640
                        # 4,4       3, 77, 4, 640*640
        xy_proj_real = self.ext_diff@xyz1_proj
        xy_proj_real = xy_proj_real[:,:,:2,:]

        return xy_proj_real
    
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
    
    #### Response function of projector
    # def cubic_interpolation(self, x_new, x_points, y_points, n):
    #     """ Interpolation function
        
    #     """
    #     tck = interpolate.splrep(x_points, y_points, k=n)   # Estimate the polynomial of nth degree by using x_points and y_points
    #     y_new = interpolate.splev(x_new, tck)
        
    #     return y_new
    
    # def compute_PRF(self):
    #     """ Compute the response function of projector

    #         return : response function of projector
            
    #     """
    #     rgb_fit = loadmat(self.rgb_fit_path)
        
    #     rgb_fit = rgb_fit['p']

    #     # gamma fitting coefficients for rgb
    #     r_a,r_b,r_c = rgb_fit[0,0],rgb_fit[0,1],rgb_fit[0,2]
    #     g_a,g_b,g_c = rgb_fit[0,3],rgb_fit[0,4],rgb_fit[0,5]
    #     b_a,b_b,b_c = rgb_fit[0,6],rgb_fit[0,7],rgb_fit[0,8]

    #     # bring projected rgb patterns
    #     rgb = torch.round(torch.linspace(20,255,10))
    #     rgb_val = torch.zeros(size=(10,3,3)) 

    #     for i in range(len(rgb)):
    #         rgb_val[i,0,0] = rgb[i]
    #         rgb_val[i,1,1] = rgb[i]
    #         rgb_val[i,2,2] = rgb[i]
        
    #     # response function
    #     wvls_bandpass = torch.linspace(400, 750, 8) # for rgb
    #     wvls_bandpass = wvls_bandpass.long()

    #     # transmission
    #     T = 1
    #     # spectralon reflectance
    #     R = 1
    #     # number of random rgb patterns
    #     N_img = 10
        
    #     # I_r,I_g, I_b 합친 I
    #     I= np.zeros(shape=(3, len(wvls_bandpass), N_img))
        
    #     # spectralon region
    #     roi = [917, 1017, 1100, 1200]
        
    #     proj_gam_RGB = np.zeros(shape=(3, N_img, 3)) #rgb 중 하나

    #     for k in range(3):
    #         if k == 0 :
    #             num = 20
    #         elif k == 1:
    #             num = 10
    #         else:
    #             num = 0
    #         for w_i, wvls_lamb in enumerate(wvls_bandpass):
    #             # camera CRF
    #             idx = (self.wvls_nm == wvls_lamb).nonzero(as_tuple=False)[0][0] 
    #             cam_CRF_value = self.camera.get_CRF()[idx,k] # specific cam CRF lambda rgb value 

    #             for i in range(N_img):
    #                 img = cv2.imread(self.rgb_result_dir + '/%dnm/calibration0/rgb_pattern_%04d.png' %(wvls_lamb, num+i), -1).astype(np.float32) 
    #                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #                 img /= 65535.0
    #                 img = img[roi[0]:roi[1], roi[2]:roi[3], :] # spectralon region
    #                 im_i_mean = img.mean(axis=(0,1)) # mean
    #                 intensity = im_i_mean[k] 

    #                 p_r, p_g, p_b = rgb_val[i, k, 0], rgb_val[i, k ,1], rgb_val[i, k, 2] 
    #                 # after gamma correction 
    #                 p_r_gamma = r_a*(p_r**r_b)+r_c
    #                 p_g_gamma = g_a*(p_g**g_b)+g_c
    #                 p_b_gamma = b_a*(p_b**b_b)+b_c

    #                 proj_gam_RGB[k,i,0],proj_gam_RGB[k,i,1], proj_gam_RGB[k,i,2] = p_r_gamma, p_g_gamma, p_b_gamma

    #                 I[k, w_i, i] = intensity/(cam_CRF_value*R*T)
                    
    #     CRF_proj = np.zeros(shape=(3, len(wvls_bandpass), 3))


    #     for k in range(3):
    #         for i in range(len(wvls_bandpass)):
    #             p = np.linalg.lstsq(proj_gam_RGB[k],I[k,i,:],rcond=None)[0]
    #             CRF_proj[k, i,:] = p
                
        
    #     # insert extra interpolation points
    #     for k in range(3):
    #         if k == 0 :
    #             CRF_proj_r_ins = np.insert(CRF_proj[k,:,k], -1, (CRF_proj[k,:,k][-1] + CRF_proj[k,:,k][-2])/2) # 725 nm insert for Red
    #             CRF_proj_r_ins = np.insert(CRF_proj_r_ins, 3, (CRF_proj[k,:,k][2] + CRF_proj[k,:,k][3])/2) # 525 nm insert for Red
    #             CRF_proj_r_ins = np.insert(CRF_proj_r_ins, 1, (CRF_proj[k,:,k][0] + CRF_proj[k,:,k][1])/2) # 425 nm insert for Red
    #             CRF_proj_r_ins = np.insert(CRF_proj_r_ins, -1, CRF_proj[k,:,k][-1]) # 780 nm insert for Red
    #         elif k == 1:
    #             CRF_proj_g_ins = np.insert(CRF_proj[k,:,k], -1, (CRF_proj[k,:,k][-1] + CRF_proj[k,:,k][-2])/2) # 725 nm insert for Red
    #             CRF_proj_g_ins = np.insert(CRF_proj_g_ins, 3, (CRF_proj[k,:,k][2] + CRF_proj[k,:,k][3])/2) # 525 nm insert for Red
    #             CRF_proj_g_ins = np.insert(CRF_proj_g_ins, 1, (CRF_proj[k,:,k][0] + CRF_proj[k,:,k][1])/2) # 425 nm insert for Red
    #             CRF_proj_g_ins = np.insert(CRF_proj_g_ins, -1, CRF_proj[k,:,k][-1]) # 780 nm insert for Red
            
    #         else:
    #             CRF_proj_b_ins = np.insert(CRF_proj[k,:,k], -1, (CRF_proj[k,:,k][-1] + CRF_proj[k,:,k][-2])/2) # 725 nm insert for Red
    #             CRF_proj_b_ins = np.insert(CRF_proj_b_ins, 3, (CRF_proj[k,:,k][2] + CRF_proj[k,:,k][3])/2) # 525 nm insert for Red
    #             CRF_proj_b_ins = np.insert(CRF_proj_b_ins, 1, (CRF_proj[k,:,k][0] + CRF_proj[k,:,k][1])/2) # 425 nm insert for Red
    #             CRF_proj_b_ins = np.insert(CRF_proj_b_ins, -1, CRF_proj[k,:,k][-1]) # 780 nm insert for Red
                
    #     # wavelength expand
    #     wvls_bandpass = wvls_bandpass.numpy()
    #     wvls_bandpass_int = np.insert(wvls_bandpass, 1, 425)
    #     wvls_bandpass_int = np.insert(wvls_bandpass_int, 4, 525)
    #     wvls_bandpass_int = np.insert(wvls_bandpass_int, len(wvls_bandpass_int)-1, 725)
    #     wvls_bandpass_int = np.insert(wvls_bandpass_int, len(wvls_bandpass_int), 780)
        
    #     # interpolation
    #     new_wvls = self.wvls
    #     wvls_bandpass_nm = wvls_bandpass_int*1e-9

    #     R_y_qe = CRF_proj_r_ins # R
    #     R_y_new = self.cubic_interpolation(new_wvls, wvls_bandpass_nm, R_y_qe , 2)

    #     G_y_qe = CRF_proj_g_ins # G
    #     G_y_new = self.cubic_interpolation(new_wvls, wvls_bandpass_nm, G_y_qe , 2)

    #     B_y_qe = CRF_proj_b_ins # B
    #     B_y_new = self.cubic_interpolation(new_wvls, wvls_bandpass_nm, B_y_qe , 2)
        
    #     CRF_proj_final = torch.zeros((self.wvls_N, 3))

    #     R_y_new = torch.tensor(R_y_new)
    #     G_y_new = torch.tensor(G_y_new)
    #     B_y_new = torch.tensor(B_y_new)

    #     CRF_proj_final[:,0] = R_y_new
    #     CRF_proj_final[:,1] = G_y_new
    #     CRF_proj_final[:,2] = B_y_new
        
    #     CRF_proj_final =np.clip(CRF_proj_final, 0, 3)
                
    #     return CRF_proj_final
    
    def get_PRF(self):
        # PRF = self.compute_PRF()
        PRF = np.load(os.path.join(self.crf_dir, 'CRF_proj.npy'))        
        
        return PRF
    
    def get_ext_proj_real(self):
        extrinsic_proj_real = self.extrinsic_proj_real()
        
        return extrinsic_proj_real
    
    def get_ext_diff(self):
        extrinsic_diff = self.extrinsic_diff()
        
        return extrinsic_diff