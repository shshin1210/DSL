import numpy as np
import os, sys
from scipy import ndimage

import torch

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.image_formation import projector
from hyper_sl.image_formation import camera

from hyper_sl.data import hdr

class HyperspectralReconstruction():
    def __init__(self, arg):
        
        # device
        self.device = "cuda:0"
        
        # arguments
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.wvls = arg.wvls
        self.date = arg.real_data_date
        self.calibrated_date = arg.calibrated_date
        
        self.new_wvls = arg.new_wvls
        self.new_wvl_num = arg.new_wvl_num

        self.depth_min, self.depth_max = arg.depth_min, arg.depth_max
        self.depth_arange = np.arange(self.depth_min *1e+3, self.depth_max*1e+3 + 1, 1)
        
        self.n_illum = arg.illum_num
        
        self.weight_D = arg.weight_D
        
        self.real_data_dir = os.path.join(arg.real_data_dir, '2023%s_real_data'%self.date)
        self.position_calibrated_data_dir = arg.position_calibrated_data_dir%self.calibrated_date
        
        # CRF, PEF, DG
        self.PEF = projector.Projector(arg, self.device).get_PEF()
        self.DG_efficiency = projector.Projector(arg, self.device).get_dg_intensity()
        self.CRF = camera.Camera(arg).get_CRF()
        
        # hdr imgs
        # self.hdr_imgs = np.load('./hdr_step5.npy')
        self.hdr_imgs = hdr.HDR(arg).make_hdr()
        np.save('./hdr_step5.npy', self.hdr_imgs)
        
    def get_depth(self):
        """
            bring gray code depth reconstructed depth values
        """
        depth = np.load(os.path.join(self.real_data_dir, './2023%s_depth.npy'%self.date))[:,:,2]*1e+3
        depth = np.round(depth).reshape(self.cam_H* self.cam_W).astype(np.int16)
        
        return depth
    
    def median_filter(self, hdr_imgs):
        """
            median filtered to hdr_imgs for more smooth ppg graph
        """
        
        # median filter
        hdr_imgs_filtered_R = np.array([ndimage.median_filter(image[:,:,0], size=4) for image in hdr_imgs])
        hdr_imgs_filtered_G = np.array([ndimage.median_filter(image[:,:,1], size=4) for image in hdr_imgs])
        hdr_imgs_filtered_B = np.array([ndimage.median_filter(image[:,:,2], size=4) for image in hdr_imgs])

        hdr_imgs_filtered = np.stack((hdr_imgs_filtered_R, hdr_imgs_filtered_G, hdr_imgs_filtered_B), axis = 3)

        # save
        np.save('./hdr_imgs_filtered.npy', hdr_imgs_filtered)
        # hdr_imgs_filtered = np.load('./hdr_imgs_filtered.npy')
        print("median filtered to hdr image")

        return hdr_imgs_filtered

    def peak_illumination_index(self, hdr_imgs):
        """
            bring calibrated first order & zero order peak illumination index
            
        """
        # zero order peak illumination index
        zero_illum_idx = np.zeros(shape=(self.cam_H * self.cam_W))
        hdr_imgs_reshape = hdr_imgs.reshape(self.n_illum, self.cam_H*self.cam_W, 3)

        for i in range(self.cam_H* self.cam_W):
            max_idx = np.argmax(hdr_imgs_reshape[:,i].mean(axis = 1))
            zero_illum_idx[i] = max_idx

        zero_illum_idx = np.round(zero_illum_idx)

        # first order peak illumination index
        first_illum_idx = np.load(os.path.join(self.position_calibrated_data_dir,'first_illum_idx_final_transp.npy'))
        first_illum_idx = first_illum_idx.reshape(self.new_wvl_num, len(self.depth_arange), self.cam_H* self.cam_W).transpose(1,0,2)
        
        return zero_illum_idx, first_illum_idx
    
    def get_mask(self, first_illum_idx):
        """
            mask out invalid red wavelengths
        """
        # NEED TO DEFINE MASK
        Mask = np.ones_like(first_illum_idx)
        Mask[first_illum_idx >= 318] = 0
        Mask[first_illum_idx < 0 ] = 0

        first_illum_idx[first_illum_idx >= 318] = 317
        first_illum_idx[first_illum_idx < 0 ] = 0
        
        return Mask, first_illum_idx
    
    def get_valid_illumination_idex(self, depth, first_illum_idx, Mask):
        """
            get the valid illumination index from 301 depth calibrated illum idx
            and mask?
            
            (valid for specific scene)
            
        """
        print("get valid illumination index for real scene")

        real_img_illum_idx = np.zeros(shape=(self.new_wvl_num, self.cam_H*self.cam_W))
        mask_idx = np.zeros(shape=(self.new_wvl_num, self.cam_H*self.cam_W))

        for i in range(self.cam_H*self.cam_W):
                if (depth[i] < 600) or (depth[i] > 900):
                        depth[i] = 600
                depth_idx = np.where(self.depth_arange == depth[i])[0][0]
                real_img_illum_idx[:,i]= first_illum_idx[depth_idx,:,i]
                mask_idx[:,i] = Mask[depth_idx,:,i]
        
        real_img_illum_idx = real_img_illum_idx.astype(np.int16).reshape(self.new_wvl_num, self.cam_H, self.cam_W)
        real_img_illum_idx_final = np.stack((real_img_illum_idx, real_img_illum_idx, real_img_illum_idx), axis = 3)

        # mask as output as well? ======================================================================================================================
        
        return real_img_illum_idx, real_img_illum_idx_final

    def dg_image_efficiency(self, zero_illum_idx, real_img_illum_idx):
        """
            diffraction grating efficiency for m = -1, 1, 0 orders for each pixels
        """
        real_img_illum_idx_reshape = real_img_illum_idx.reshape(self.new_wvl_num, self.cam_H*self.cam_W)
        
        # DG efficiency for all pixels
        DG_efficiency_image = np.zeros(shape=(self.cam_H * self.cam_W, self.new_wvl_num))

        for i in range(self.cam_H * self.cam_W):
            if zero_illum_idx[i] > real_img_illum_idx_reshape[0,i]: # 430nm # -1 order
                DG_efficiency_image[i,:] =  self.DG_efficiency[0]
            elif zero_illum_idx[i] < real_img_illum_idx_reshape[0,i]: # +1 order
                DG_efficiency_image[i,:] =  self.DG_efficiency[2]
            else: # else
                DG_efficiency_image[i,:] = 0

        return DG_efficiency_image
        
    def SVD(self):
        
        # datas        
        hdr_imgs = self.median_filter(self.hdr_imgs / 65535.)
        np.save('./hdr_imgs_filtered_%s.npy'%self.date, hdr_imgs)
        # hdr_imgs = np.load('./hdr_imgs_filtered.npy')
        
        zero_illum_idx, first_illum_idx = self.peak_illumination_index(hdr_imgs)
        mask, first_illum_idx = self.get_mask(first_illum_idx)
        
        depth = self.get_depth()
        real_img_illum_idx, real_img_illum_idx_final = self.get_valid_illumination_idex(depth, first_illum_idx, mask)
        
        DG_efficiency_image = self.dg_image_efficiency(zero_illum_idx, real_img_illum_idx)
        
        print("Calculate SVD...")
        
        x, y, z = np.meshgrid(np.arange(580), np.arange(890), np.arange(3), indexing='ij')

        # D matrix
        i_mat = np.eye(self.new_wvl_num)
        diagonal_indices = np.diag_indices(i_mat.shape[0])
        new_diagonal_indices_col = np.copy(diagonal_indices[1])
        new_diagonal_indices_col[:-1] = diagonal_indices[1][:-1] + 1
        i_mat[(diagonal_indices[0], new_diagonal_indices_col)] = -1
        D = i_mat
        D[-1] = i_mat[-2]

        D = torch.tensor(D, device=self.device)

        # to tensor, device
        CRF = torch.tensor(self.CRF, device=self.device).type(torch.float32)
        PEF = torch.tensor(self.PEF, device=self.device).type(torch.float32)
        hdr_imgs_t = torch.tensor(hdr_imgs, device=self.device)
        first_illum_idx_final_transp_t = torch.tensor(real_img_illum_idx_final, device=self.device)
        DG_efficiency_image_t = torch.tensor(DG_efficiency_image.reshape(self.cam_H* self.cam_W, -1), device= self.device) # H x W, wvls

        # pattern
        white_patt = torch.ones(size = (self.cam_H * self.cam_W, 3), device=self.device) * 0.8
        white_patt_hyp = white_patt @ PEF.T
        white_patt_hyp = white_patt_hyp.squeeze()

        CRF_sum = torch.tensor(CRF, device=self.device).sum(axis = 1)

        total_hyp_ref = []

        # summation of Image RGB channel
        I_C = hdr_imgs_t[first_illum_idx_final_transp_t.long(), x, y, z].permute(1, 2, 0, 3).sum(axis = 3).reshape(-1, self.new_wvl_num, 1) # H x W, wvls, 1
        A = (CRF_sum.unsqueeze(dim = 0) * white_patt_hyp * DG_efficiency_image_t).unsqueeze(dim =2) # HxW, wvls, 1

        A_diag = torch.diag_embed((A*A).squeeze())

        weight_D_DT = (self.weight_D*D.T@D).unsqueeze(dim = 0)

        csj_x = torch.linalg.solve(A_diag + weight_D_DT, A*I_C.reshape(self.cam_H*self.cam_W, self.new_wvl_num,1))

        total_hyp_ref = csj_x.squeeze()
        total_hyp_ref = total_hyp_ref.reshape(self.cam_H, self.cam_W, self.new_wvl_num)

        total_hyp_ref = np.load('./total_hyp_ref_%s.npy'%self.date)
        # np.save('./total_hyp_ref_%s.npy'%self.date, total_hyp_ref.detach().cpu().numpy())        
        return total_hyp_ref
    
if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
    total_hyp_ref = HyperspectralReconstruction(arg).SVD()
    
    print('test') 