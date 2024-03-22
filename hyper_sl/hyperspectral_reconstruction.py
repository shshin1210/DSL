import numpy as np
import os, sys, cv2, random
from scipy import ndimage

import torch

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.image_formation import projector
from hyper_sl.image_formation import camera

from hyper_sl.data import hdr

class HyperspectralReconstruction():
    """
        Reconstruct hyperspectral iamge

        Arguments
            - wvls: linespace of minimum wavelengths to maximum wavelength in 10 nm interval(430nm to 660nm in 10nm intervals)
            - calibrated date: date of calibration of correspondence model
            - new_wvls: linespace of minimum wavelengths to maximum wavelength in 5nm interval (430nm to 660nm in 5nm intervals)
            - n_illum: number of illumination
            
            Calibrated parameters
            - PEF: projector emission function
            - CRF: camera response function
            - DG_efficiency: diffraction grating efficiency 

            Real HDR data
            - hdr_imgs: captured scene into HDR image
    """
    def __init__(self, arg):
        
        # device
        self.arg = arg
        self.device = "cuda:0"
        self.epoch = arg.epoch_num
        
        # arguments
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.wvls = arg.wvls
        
        self.new_wvls = arg.new_wvls
        self.new_wvl_num = arg.new_wvl_num

        self.depth_min, self.depth_max = arg.depth_min, arg.depth_max
        self.depth_arange = np.arange(self.depth_min *1e+3, self.depth_max*1e+3 + 1, 1)
        
        self.n_illum = arg.illum_num
                
        # calibrated parameters
        self.PEF = projector.Projector(arg, self.device).get_PEF()
        self.CRF = camera.Camera(arg).get_CRF()
        self.DG_efficiency = projector.Projector(arg, self.device).get_dg_intensity()
        
        # directory
        self.position_calibrated_data_dir = arg.position_calibrated_data_dir
        self.path_to_depth = arg.path_to_depth
        
        # hdr imgs
        self.hdr_imgs = hdr.HDR(arg).make_hdr()
        
    def get_depth(self):
        """
            bring gray code depth reconstructed depth values
            
            Output
                - depth: depth with unit mm
                
        """
        
        depth = np.load(self.path_to_depth)[:,:,2]*1e+3
        depth = np.round(depth).reshape(self.cam_H* self.cam_W).astype(np.int16)
        
        return depth
    
    def median_filter(self, hdr_imgs):
        """
            median filtered to hdr_imgs for more smooth ppg graph
            
            Input
                - hdr_imgs: HDR image
                
            Output
                - hdr_imgs_filtered: 
        """
        
        # median filter
        hdr_imgs_filtered_R = np.array([ndimage.median_filter(image[:,:,0], size=4) for image in hdr_imgs])
        hdr_imgs_filtered_G = np.array([ndimage.median_filter(image[:,:,1], size=4) for image in hdr_imgs])
        hdr_imgs_filtered_B = np.array([ndimage.median_filter(image[:,:,2], size=4) for image in hdr_imgs])

        hdr_imgs_filtered = np.stack((hdr_imgs_filtered_R, hdr_imgs_filtered_G, hdr_imgs_filtered_B), axis = 3)

        return hdr_imgs_filtered

    def peak_illumination_index(self, hdr_imgs):
        """
            bring calibrated first order & zero order peak correspondence model
            
            Input
                - hdr_imgs : Real captured HDR image for zero order correspondence model
            
            Output
                - zero and first order correspondence model
        """
        # zero order peak correspondence model
        zero_illum_idx = np.zeros(shape=(self.cam_H * self.cam_W))
        hdr_imgs_reshape = hdr_imgs.reshape(self.n_illum, self.cam_H*self.cam_W, 3)

        for i in range(self.cam_H* self.cam_W):
            max_idx = np.argmax(hdr_imgs_reshape[:,i].mean(axis = 1))
            zero_illum_idx[i] = max_idx

        zero_illum_idx = np.round(zero_illum_idx)
        zero_illum_idx = np.repeat(zero_illum_idx.reshape(1, 1, self.cam_H*self.cam_W), len(self.depth_arange), axis=0)

        # first order peak correspondence model
        first_illum_idx = np.load(self.position_calibrated_data_dir)
        first_illum_idx = first_illum_idx.reshape(self.new_wvl_num, len(self.depth_arange), self.cam_H* self.cam_W).transpose(1,0,2)
        
        return zero_illum_idx, first_illum_idx
    
    def get_mask(self, first_illum_idx):
        """
            mask out invalid red wavelengths
            
            Input
                - first_illum_idx : first order correspondence model
                
            Output
                - Mask for invalid wavelengths
                - first order correspondence model
            
        """
        
        # Need to define mask
        Mask = np.ones_like(first_illum_idx)
        Mask[first_illum_idx >= self.n_illum] = 0
        Mask[first_illum_idx < 0 ] = 0

        first_illum_idx[first_illum_idx >= self.n_illum] = self.n_illum -1
        first_illum_idx[first_illum_idx < 0 ] = 0
        
        return Mask, first_illum_idx
    
    def get_valid_illumination_idex(self, depth, first_illum_idx, Mask):
        """
            get the scene dependent correspondence model calibrated correspondence model

            Input
                - depth : depth of the scene
                - first_illum_idx : first order correspondence model
                - Mask : Mask for invalid/valid wavelengths
            
        """

        real_img_illum_idx = np.zeros(shape=(self.new_wvl_num, self.cam_H*self.cam_W))
        mask_idx = np.zeros(shape=(self.new_wvl_num, self.cam_H*self.cam_W))

        for i in range(self.cam_H*self.cam_W):
                # filter out depth out of range
                if (depth[i] < self.depth_min) or (depth[i] > self.depth_max):
                        depth[i] = self.depth_min
                depth_idx = np.where(self.depth_arange == depth[i])[0][0]
                real_img_illum_idx[:,i]= first_illum_idx[depth_idx,:,i]
                mask_idx[:,i] = Mask[depth_idx,:,i]
        
        real_img_illum_idx = real_img_illum_idx.astype(np.int16).reshape(self.new_wvl_num, self.cam_H, self.cam_W)
        real_img_illum_idx_final = np.stack((real_img_illum_idx, real_img_illum_idx, real_img_illum_idx), axis = 3)
        
        return real_img_illum_idx, real_img_illum_idx_final, mask_idx

    def dg_image_efficiency(self, zero_illum_idx, real_img_illum_idx):
        """
            diffraction grating efficiency for m = -1, 1, 0 orders for each pixels
            
            Input
                - zero_illum_idx : zero order correspondence model
                - real_img_illum_idx : scene dependent correspondence model
                
            Output
                - DG_efficiency_image : scene dependent diffraction grating efficiency
            
        """
        real_img_illum_idx_reshape = real_img_illum_idx.reshape(self.new_wvl_num, self.cam_H*self.cam_W)
        
        # DG efficiency for all pixels
        DG_efficiency_image_first = np.zeros(shape=(self.cam_H * self.cam_W, self.new_wvl_num))
        DG_efficiency_image_zero = np.ones(shape=(self.cam_H * self.cam_W, self.new_wvl_num))

        for i in range(self.cam_H * self.cam_W):
            if zero_illum_idx[i] > real_img_illum_idx_reshape[0,i]: # 430nm # -1 order
                DG_efficiency_image_first[i,:] =  self.DG_efficiency[0]
                
            elif zero_illum_idx[i] < real_img_illum_idx_reshape[0,i]: # +1 order
                DG_efficiency_image_first[i,:] =  self.DG_efficiency[2]
            
            else: # else
                DG_efficiency_image_first[i,:] = 0

        return DG_efficiency_image_zero, DG_efficiency_image_first
        
    def Optimization(self):
        
        # datas        
        hdr_imgs = self.median_filter(self.hdr_imgs / 65535.)
        
        # correspondence model
        zero_illum_idx, first_illum_idx = self.peak_illumination_index(hdr_imgs)
        mask, first_illum_idx = self.get_mask(first_illum_idx)
        real_img_illum_idx, real_img_illum_idx_final, mask_idx = self.get_valid_illumination_idex(depth, first_illum_idx, mask)
        DG_efficiency_image_zero, DG_efficiency_image_first = self.dg_image_efficiency(zero_illum_idx, real_img_illum_idx)

        # depth
        depth = self.get_depth()
        
        # GT RGB image
        x, y, z = np.meshgrid(np.arange(self.cam_H), np.arange(self.cam_W), np.arange(3), indexing='ij')
        GT_I_RGB_FIRST = hdr_imgs[real_img_illum_idx_final, x, y, z].transpose(1, 2, 0, 3)   
        GT_I_RGB_ZERO =  hdr_imgs[zero_illum_idx, x, y, z].transpose(1, 2, 0, 3)
        
        # Mask
        # Make all wavelength invalid if one wavelength is invalid
        for i in range(self.cam_H* self.cam_W):
            if 0 in mask_idx[:,i]:
                mask_idx[:,i] = 0
                
        mask = np.zeros((self.cam_H, self.cam_W, self.new_wvl_num, 1))
        mask_temp = mask_idx.T.reshape(self.cam_H, self.cam_W, self.new_wvl_num, 1)
        for i in range(self.new_wvl_num):
            a = mask_temp[:,:,i,:]
            sigma = self.arg.sigma

            dst = cv2.GaussianBlur(a, (0,0), sigma)
            mask[:,:,i,:] = torch.tensor(dst[...,None])
        
        # Optimization
        print("Start Optimization")
        epoch = self.epoch
        losses = [] 

        # learning rate & decay step
        lr = self.arg.lr
        decay_step = self.arg.decay_step
        gamma = self.arg.gamma

        # optimized paramter (CRF & PEF)
        initial_value = torch.ones(size =(self.cam_H*self.cam_W, self.new_wvl_num))/2
        initial_value = torch.logit(initial_value)
        _opt_param =  torch.tensor(initial_value, dtype= torch.float, requires_grad=True, device= self.device)

        # optimizer and schedular
        optimizer = torch.optim.Adam([_opt_param], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = gamma)

        # shape : 3, num new wvls
        PEF_dev = torch.tensor(self.PEF, dtype = torch.float).to(self.device).T
        CRF_dev = torch.tensor(self.CRF, dtype = torch.float).to(self.device)
        DG_efficiency_first_dev = torch.tensor(DG_efficiency_image_first.reshape(self.cam_H*self.cam_W, -1), device= self.device)
        DG_efficiency_zero_dev = torch.tensor(DG_efficiency_image_zero.reshape(self.cam_H*self.cam_W, -1), device= self.device)

        # depth scalar
        A =  self.arg.depth_scalar
        depth_scalar = ((depth.astype(np.int32))**2) / A

        depth_scalar_dev = torch.tensor(depth_scalar, dtype = torch.float).to(self.device).T
        unreached_mask_tensor = torch.tensor(mask_idx.T, device=self.device).reshape(self.cam_H, self.cam_W, self.num, 1)

        # white pattern into multi-spectral channels
        white_patt = torch.ones(size = (self.cam_H * self.cam_W, 3), device=self.device) * 0.8
        white_patt_hyp = white_patt @ PEF_dev
        white_patt_hyp = white_patt_hyp.squeeze()

        GT_I_RGB_FIRST_tensor = torch.tensor(GT_I_RGB_FIRST, device=self.device)
        GT_I_RGB_ZERO_tensor = torch.tensor(GT_I_RGB_ZERO, device=self.device)

        weight_first = self.arg.weight_first
        weight_zero = self.arg.weight_zero
        weight_unreach = self.arg.weight_unreach
        weight_spectral = self.arg.weight_spectral

        loss_vis = []
        A_first = CRF_dev.unsqueeze(dim = 0) * white_patt_hyp.unsqueeze(dim = -1) * DG_efficiency_first_dev.unsqueeze(dim = -1)
        A_zero = CRF_dev.unsqueeze(dim = 0) * white_patt_hyp.unsqueeze(dim = -1) * DG_efficiency_zero_dev.unsqueeze(dim = -1)

        for i in range(epoch):
            # initial loss
            loss = 0

            opt_param = torch.sigmoid(_opt_param)

            Simulated_I_RGB_first = opt_param.unsqueeze(dim = -1) * A_first / depth_scalar_dev.unsqueeze(dim = -1).unsqueeze(dim = -1) 
            Simulated_I_RGB_zero = torch.sum(opt_param.unsqueeze(dim = -1) * A_zero / depth_scalar_dev.unsqueeze(dim = -1).unsqueeze(dim = -1), axis=1) 
            
            image_loss_first = torch.abs((Simulated_I_RGB_first.reshape(self.cam_H, self.cam_W, self.new_wvl_num, 3)) - GT_I_RGB_FIRST_tensor) * unreached_mask_tensor / (self.cam_H*self.cam_W)
            loss += weight_first * image_loss_first.sum()

            unreached_loss = torch.abs((Simulated_I_RGB_zero.reshape(self.cam_H, self.cam_W, 1, 3)) - GT_I_RGB_ZERO_tensor) * (1 - unreached_mask_tensor) / (self.cam_H*self.cam_W)
            image_loss_zero = torch.abs((Simulated_I_RGB_zero.reshape(self.cam_H, self.cam_W, 1, 3)) - GT_I_RGB_ZERO_tensor) / (self.cam_H*self.cam_W)
            
            loss += (weight_zero * image_loss_zero.sum() + weight_unreach*unreached_loss.sum())
            loss +=  weight_unreach*unreached_loss.sum()
            
            hyp_dL2 = ((opt_param[:,:-1] - opt_param[:,1:])**2).sum()/(self.cam_H*self.cam_W)

            loss += weight_spectral*(hyp_dL2)

            loss = loss.sum()
            loss_vis.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
                
            if i % 100 == 0:
                print(f"Epoch : {i}/{epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
        
        return opt_param

if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
    Reconstructed_hyperspectral_relfectance = HyperspectralReconstruction(arg).Optimization()