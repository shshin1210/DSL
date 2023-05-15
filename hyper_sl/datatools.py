import os
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

import sys

# sys.path.append('/home/shshin/Scalable-Hyperspectral-3D-Imaging')

from hyper_sl.data import create_data_patch
from hyper_sl.image_formation.etc import hyp_renderer

#  pixel-wise totally random with random scene
class pixelData(Dataset):
    """
        Arguments :
        
            - Train (True/ False) : training mode / validation mode
            - pixel_num : # of pixels to render (total pixel / epoch)
            - random (True/ False) : random camera plane coordinate / unrandom camera plane coordinate
            - bring data (True/ False) : Use saved data / Use new data and render it now
            
        Render total resolution pixels for every scene with random hyperspectral reflectance and depth and fixed camera coordinate
        
        Returns random # of pixels of rendered result 
                normalized projector plane coord (ground truth)
                normalized camera plane coord

    """    
    def __init__(self, arg, train = True, eval = False, pixel_num = 0, random = True, real = False):

        self.arg = arg

        # bring class
        # self.create_data = create_data.createData
        self.create_data = create_data_patch.createData
        self.render = hyp_renderer
        
        # arguments
        self.pixel_num = pixel_num  # total pixel
        self.train = train
        self.eval = eval
        self.real = real
        self.random = random

    def __len__(self):
        if self.train == True and self.eval == False:
            return self.arg.scene_train_num # training scene의 개수
        elif self.train == False and self.eval == False:
            return self.arg.scene_test_num # testing scene의 개수
        elif self.train == False and self.eval == True:
            return self.arg.scene_eval_num
        elif self.real == True:
            return self.arg.scene_real_num
        
    def __getitem__(self, index):
        # training / validation set
        if self.train == True and self.eval == False:
            depth = self.create_data(self.arg, "depth", self.pixel_num, random = self.random).create()
            normal = self.create_data(self.arg, "normal", self.pixel_num, random = self.random).create()
            hyp = self.create_data(self.arg, 'hyp', self.pixel_num, random = self.random).create()
            occ = self.create_data(self.arg, 'occ', self.pixel_num, random = self.random).create()
            cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()
            
            return [depth, normal, hyp, occ, cam_coord]
        
        elif self.train == False and self.eval == False:
            depth = self.create_data(self.arg, "depth", self.pixel_num, random = self.random).create()
            normal = self.create_data(self.arg, "normal", self.pixel_num, random = self.random).create()
            hyp = self.create_data(self.arg, 'hyp', self.pixel_num, random = self.random).create()
            occ = self.create_data(self.arg, 'occ', self.pixel_num, random = self.random).create()
            cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()

            return [depth, normal, hyp, occ, cam_coord]
        
        # evaluation set
        elif self.train == False and self.eval == True: # evaluation
            depth = self.create_data(self.arg, "depth", self.pixel_num, random = self.random, i = index).create()
            normal = self.create_data(self.arg, "normal", self.pixel_num, random = self.random, i = index).create()
            hyp = self.create_data(self.arg, 'hyp', self.pixel_num, random = self.random, i = index).create()
            occ = self.create_data(self.arg, 'occ', self.pixel_num, random = self.random, i = index).create()
            cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()

            return [depth, normal, hyp, occ, cam_coord]

        elif self.real == True:
            N3_arr, illum_data = self.create_data(self.arg, 'real', self.pixel_num, self.random, i = index).create()
            # cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()

            return [N3_arr, illum_data, cam_coord]
            # return [cam_coord]
            
            
if __name__ == "__main__":
    
    from hyper_sl.utils.ArgParser import Argument
    from torch.utils.data import DataLoader
 
    argument = Argument()
    arg = argument.parse()
    
    
    pixel_num = 640*640
    
    eval_dataset = pixelData(arg, train = False, eval = True, pixel_num = 640*640, bring_data = False, random = False)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)
    
    for i, data in enumerate(eval_loader):
        N3_arr = data[0] # B, # pixel, N, 3
        gt_xy = data[1] # B, # pixel, 2
        
        # reshape
        N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
        N3_arr = N3_arr.unsqueeze(dim = 1) 
        gt_xy = gt_xy.reshape(-1,2)
        
        # normalization of N3_arr
        
        N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
        N3_arr_max = N3_arr_r.max(axis = 1)
            
        N3_arr_max = N3_arr.max(axis=2).values[:,None,:]
        N3_arr_min = N3_arr.min(axis=2).values[:,None,:]
        N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)
        
        N3_arr_normalized[torch.isnan(N3_arr_normalized)] = 0.
        
        N3_arr_normalized_vis = N3_arr_normalized.detach().cpu().numpy()
    
    print('end')