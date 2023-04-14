import os
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

import sys

# sys.path.append('/home/shshin/Scalable-Hyperspectral-3D-Imaging')

from hyper_sl.data import create_data
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
    def __init__(self, arg, train = True, eval = False, pixel_num = 0, random = True):

        self.arg = arg

        # bring class
        self.create_data = create_data.createData
        self.render = hyp_renderer
        
        # arguments
        self.pixel_num = pixel_num  # total pixel
        self.train = train
        self.eval = eval
        self.random = random

    def __len__(self):
        if self.train == True and self.eval == False:
            return self.arg.scene_train_num # training scene의 개수
        elif self.train == False and self.eval == False:
            return self.arg.scene_test_num # testing scene의 개수
        else:
            return self.arg.eval_num
        
    def __getitem__(self, index):
        # training / validation set
        if self.train == True and self.eval == False:
            depth = self.create_data(self.arg, "depth", self.pixel_num, random = self.random).create()
            normal = self.create_data(self.arg, "normal", self.pixel_num, random = self.random).create()
            hyp = self.create_data(self.arg, 'hyp', self.pixel_num, random = self.random).create()
            occ = self.create_data(self.arg, 'occ', self.pixel_num, random = self.random).create()
            cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()
            
            return [depth, normal, hyp, occ, cam_coord]
        
        if self.train == False and self.eval == False:
            depth = self.create_data(self.arg, "depth", self.pixel_num, random = self.random).create()
            normal = self.create_data(self.arg, "normal", self.pixel_num, random = self.random).create()
            hyp = self.create_data(self.arg, 'hyp', self.pixel_num, random = self.random).create()
            occ = self.create_data(self.arg, 'occ', self.pixel_num, random = self.random).create()
            cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()

            return [depth, normal, hyp, occ, cam_coord]
        
        # evaluation set
        else: # evaluation
            depth = self.create_data(self.arg, "depth", self.pixel_num, random = self.random, i = index).create()
            normal = self.create_data(self.arg, "normal", self.pixel_num, random = self.random, i = index).create()
            hyp = self.create_data(self.arg, 'hyp', self.pixel_num, random = self.random, i = index).create()
            occ = self.create_data(self.arg, 'occ', self.pixel_num, random = self.random, i = index).create()
            cam_coord = self.create_data(self.arg, 'coord', self.pixel_num, random = self.random).create()

            return [depth, normal, hyp, occ, cam_coord]

#  pixel-wise totally random with static scene
class pixelData2(Dataset):
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
    def __init__(self, arg, train = True, eval = False, pixel_num = 0, random = True):

        self.arg = arg

        # bring class
        self.create_data = create_data.createData
        self.render = hyp_renderer
        
        # arguments
        self.pixel_num = pixel_num  # total pixel
        self.train = train
        self.eval = eval
        self.random = random
        self.wvl_num = arg.wvl_num
        self.cam_resolution = arg.cam_H
        self.pixel_num = pixel_num # # of pixels to pick for training set
        
        # training set
        if (self.train == True) and (self.eval == False):
            self.scene_num = arg.scene_train_num
            self.depth_data, self.normal_data, self.hyp_data, self.occ_data, self.cam_coord_data = self.createDataset(self.scene_num, self.random)
            
        # validation set
        elif (self.train == False) and (self.eval == False):
            self.scene_num = arg.scene_test_num
            self.depth_data, self.normal_data, self.hyp_data, self.occ_data, self.cam_coord_data = self.createDataset(self.scene_num, self.random)
               
        # evaluation set
        elif (self.train == False) and (self.eval == True):
            self.scene_num = arg.eval_num
            self.depth_data, self.normal_data, self.hyp_data, self.occ_data, self.cam_coord_data = self.createDataset(self.scene_num, self.random)
    
    
    def __len__(self):
        if self.train == True and self.eval == False:
            return self.arg.scene_train_num # training scene의 개수
        elif self.train == False and self.eval == False:
            return self.arg.scene_test_num # testing scene의 개수
        else:
            return self.arg.eval_num
        
    def __getitem__(self, index):
        # training / validation set
        if self.eval == False:
            index_rand = np.random.randint(0, self.arg.cam_W * self.arg.cam_H, size=(self.pixel_num))
        else: # evaluation
            index_rand = np.arange(0, self.arg.cam_W * self.arg.cam_H)
            return [self.depth_data[index, index_rand,...], self.normal_data[index, :, index_rand], self.hyp_data[index, index_rand,...], self.occ_data[index, index_rand,...], self.cam_coord_data[index, index_rand,...]]
        
        return [self.depth_data[index, index_rand,...], self.normal_data[index, :, index_rand], self.hyp_data[index, index_rand,...], self.occ_data[index, index_rand,...], self.cam_coord_data[index, index_rand,...]]

    def createDataset(self, n_scene, random):
        pixel_num = self.cam_resolution*self.cam_resolution
        
        depth_data = torch.zeros(size=(n_scene, pixel_num))
        normal_data = torch.zeros(size=(n_scene, 3, pixel_num))
        hyp_data = torch.zeros(size=(n_scene, pixel_num, self.wvl_num))
        occ_data = torch.zeros(size=(n_scene, pixel_num))
        cam_coord_data = torch.zeros(size=(n_scene, pixel_num, 3))
        
        for i in range(n_scene):
            print(f'Creating {i}-th data...')
            depth = self.create_data(self.arg, "depth", pixel_num, random = random, i = i).create()
            normal = self.create_data(self.arg, "normal", pixel_num, random = random, i = i).create()
            hyp = self.create_data(self.arg, 'hyp', pixel_num, random = random, i = i).create()
            occ = self.create_data(self.arg, 'occ', pixel_num, random = random, i = i).create()
            cam_coord = self.create_data(self.arg, 'coord', pixel_num, random = random).create()
            
            depth_data[i] = depth
            normal_data[i] = normal
            hyp_data[i] = hyp
            occ_data[i] = occ
            cam_coord_data[i] =cam_coord
            
        return [depth_data, normal_data, hyp_data, occ_data, cam_coord_data]

# create scene first
class hypData(Dataset):
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
    def __init__(self, arg, train = True, eval = False, pixel_num = 0, bring_data = False, random = True):

        self.arg = arg

        # bring class
        self.create_data = create_data
        self.render = hyp_renderer
        
        # arguments
        self.pixel_num = arg.cam_W * arg.cam_H  # total pixel
        self.train = train
        self.bring_data = bring_data
        self.random = random
        self.eval = eval
        
        self.train_pixel = pixel_num # # of pixels to pick for training set
        self.test_pixel = pixel_num # # of pixels to pick for training set
        
        # class
        self.hyp_renderer = hyp_renderer.PixelRenderer(self.arg)
        
        # training set
        if (self.train == True) and (self.eval == False):
            self.scene_num = arg.scene_train_num
            if self.bring_data == True:
                self.scene_data = torch.load(os.path.join(arg.random_pixel_train_dir, arg.random_pixel_scene_fn))
                self.xy_proj_data = torch.load(os.path.join(arg.random_pixel_train_dir, arg.random_pixel_xyproj_fn))
                self.hyp_gt = torch.load(os.path.join(arg.random_pixel_train_dir, arg.random_pixel_hyp_gt))
            else:
                self.scene_data, self.xy_proj_data, _, self.hyp_gt = self.createDataset(self.pixel_num, self.scene_num, self.random, self.eval)
                if not os.path.exists(arg.random_pixel_train_dir):
                    os.mkdir(arg.random_pixel_train_dir)
                torch.save(self.scene_data, os.path.join(arg.random_pixel_train_dir, arg.random_pixel_scene_fn))
                torch.save(self.xy_proj_data, os.path.join(arg.random_pixel_train_dir, arg.random_pixel_xyproj_fn))
                torch.save(self.hyp_gt, os.path.join(arg.random_pixel_train_dir, arg.random_pixel_hyp_gt))
                
        # validation set
        elif (self.train == False) and (self.eval == False):
            self.scene_num = arg.scene_test_num
            if self.bring_data == True:
                self.scene_data = torch.load(os.path.join(arg.random_pixel_val_dir, arg.random_pixel_scene_fn))
                self.xy_proj_data = torch.load(os.path.join(arg.random_pixel_val_dir, arg.random_pixel_xyproj_fn))
                self.hyp_gt = torch.load(os.path.join(arg.random_pixel_val_dir, arg.random_pixel_hyp_gt))
            
            else:
                self.scene_data, self.xy_proj_data, _, self.hyp_gt = self.createDataset(self.pixel_num, self.scene_num, self.random, self.eval)
                
                if not os.path.exists(arg.random_pixel_val_dir):
                    os.mkdir(arg.random_pixel_val_dir)
                torch.save(self.scene_data, os.path.join(arg.random_pixel_val_dir, arg.random_pixel_scene_fn))
                torch.save(self.xy_proj_data, os.path.join(arg.random_pixel_val_dir, arg.random_pixel_xyproj_fn))
                torch.save(self.hyp_gt, os.path.join(arg.random_pixel_val_dir, arg.random_pixel_hyp_gt))
                
        # evaluation set
        elif (self.train == False) and (self.eval == True):
            self.scene_num = arg.eval_num
            if self.bring_data == True:
                self.scene_data = torch.load(os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_scene_fn))
                self.xy_proj_data = torch.load(os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_xyproj_fn))
                self.xy_proj_real = torch.load(os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_xy_real_fn))
                self.hyp_gt = torch.load(os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_hyp_gt))
            else:
                self.scene_data, self.xy_proj_data, self.xy_proj_real, self.hyp_gt = self.createDataset(self.pixel_num, self.scene_num, self.random, self.eval)
                
                if not os.path.exists(arg.random_pixel_eval_dir):
                    os.mkdir(arg.random_pixel_eval_dir)

                torch.save(self.scene_data, os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_scene_fn))
                torch.save(self.xy_proj_data, os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_xyproj_fn))
                torch.save(self.xy_proj_real, os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_xy_real_fn))
                torch.save(self.hyp_gt, os.path.join(arg.random_pixel_eval_dir, arg.random_pixel_hyp_gt))
                
    def __len__(self):
        if self.train == True and self.eval == False:
            return self.arg.scene_train_num # training scene의 개수
        elif self.train == False and self.eval == False:
            return self.arg.scene_test_num # testing scene의 개수
        else:
            return self.arg.eval_num
        
    def __getitem__(self, index):
        if self.train == True and self.eval == False:
            index_rand = np.random.randint(0, self.arg.cam_W * self.arg.cam_H, size=(self.train_pixel))
        elif self.train == False and self.eval == False:
            index_rand = np.random.randint(0, self.arg.cam_W * self.arg.cam_H, size=(self.test_pixel))
        # evaluation
        else:
            index_rand = np.arange(0, self.arg.cam_W * self.arg.cam_H)
            return [self.scene_data[index, index_rand, ...], self.xy_proj_data[index, index_rand, ...], self.xy_proj_real[index, index_rand,... ], self.hyp_gt[index, index_rand,...]]
        
        return [self.scene_data[index, index_rand, ...], self.xy_proj_data[index, index_rand, ...], self.hyp_gt[index, index_rand,...]]
        
    def createDataset(self, pixel_num, N_scene, random, eval):
        
        scene_data, xy_proj_data, xy_proj_real, hyp_gt = self.hyp_renderer.render(n_scene = N_scene, random = random, pixel_num = pixel_num, eval = eval)

        return [scene_data, xy_proj_data, xy_proj_real,hyp_gt]

    
if __name__ == "__main__":
    
    from hyper_sl.utils.ArgParser import Argument
    from torch.utils.data import DataLoader
 
    argument = Argument()
    arg = argument.parse()
    
    
    pixel_num = 640*640
    
    # 기존 rendering 방법 =================================================================================================================
    # 기존 방법으로 얻은 xy proj plane gt
    # xy_gt = HypDepthData(arg).xy_proj()
    # cam_N_img = HypDepthData(arg).makeN3Arr()
    
    
    # new rendering  방법 =================================================================================================================
    # new rendering 방법으로 (기존 dataset 이용하여) 얻은 proj plane gt, 3d xyz points
    # img_hyp_text_dir = "C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/img_hyp_text"
    # img_hyp_file = "img_hyp_text_000.npy"

    # depth = hyp_create_data.createData(arg, "depth", pixel_num, random = False, i = 0).create()
    # normal = hyp_create_data.createData(arg, "normal", pixel_num, random = False, i = 0).create()
    # hyp = hyp_create_data.createData(arg, 'hyp', pixel_num, random = False, i = 0).create()
    # occ = hyp_create_data.createData(arg, 'occ', pixel_num, random = False, i = 0).create()
    # # cam_coord = hyp_create_data.createData(arg, 'coord', pixel_num, random = True).create()

    # # 1 scene에 대해 N개의 pattern을 rendering 해주는 함수
    # N3Arr, xy_proj = hyp_pixel_renderer.PixelRenderer(arg).render(arg.illum_num, depth=depth, normal = normal, hyp= hyp, occ = occ, pixel_num = pixel_num)  

    
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