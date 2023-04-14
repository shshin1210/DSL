import numpy as np
import torch, os
import scipy.io as sci

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils.load_data import load_data

class createData():
    def __init__(self, arg, data_type, pixel_num, random = True, i = 0):
        
        # dir
        self.mat_dir = arg.img_hyp_texture_dir
        self.output_dir = arg.output_dir
        
        # arguments
        self.data_type = data_type
        self.pixel_num = pixel_num
        self.random = random
        self.i = i
        self.arg = arg
        
        # params
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.cam_focal_length = arg.focal_length * 1e-3
        self.cam_sensor_width = arg.sensor_width *1e-3
        self.wvls_n = arg.wvl_num
        
        # class
        self.load_data = load_data(arg)
        
        
    def create(self):
        if self.data_type == "depth":
            depth = self.createDepth(self.pixel_num, self.random, self.i)
            return depth
        
        elif self.data_type == "hyp":
            hyp = self.createHyp(self.pixel_num, self.random, self.i)
            return hyp
        
        elif self.data_type == "occ":
            occ = self.createOcclusion(self.pixel_num, self.random, self.i)
            return occ
        
        elif self.data_type == 'coord':
            cam_coord = self.createCamcoord(self.pixel_num, self.random)
            return cam_coord

        elif self.data_type == 'normal':
            normal = self.createNormal(self.pixel_num, self.random, self.i)
            
            return normal
        
    def createDepth(self, pixel_num, random, i):
        """ Create Depth for each pixels

            pixel_num : # of pixels to create random depth data
            
            returns : torch.tensor ( # pixel, )
        """
        if random == True:
            # meter 단위 depth
            depth_min = 1.
            depth_max = 1.4
            
            depth = (depth_max - depth_min)* torch.rand((pixel_num)) + depth_min
            
        else:
            depth = self.load_data.load_depth(i)
            depth = depth.reshape(self.cam_W * self.cam_H)
            
        return depth
    
    def createOcclusion(self, pixel_num, random, i):
        """ Create Depth for each pixels

            pixel_num : # of pixels to create random depth data
            
            returns : torch.tensor ( # pixel, )
        """
        if random == True:
            # no occlusions in random dataset
            occ = torch.ones(size = (pixel_num, ))
            
        else:
            occ = self.load_data.load_occ(i)
            occ = occ.reshape(self.cam_W* self.cam_H)
            
        return occ
            
    def createNormal(self, pixel_num, random, i):
        """ Create Noraml for each pixels

            pixel_num : # of pixels to create random Noraml data
            
            returns : torch.tensor ( # pixel, )
        """
        if random == True:
            normal = torch.zeros(size=(3, pixel_num))
            
            # normal
            normal_min = 0.001
            normal_max = 0.9999
            
            normal = (normal_max - normal_min) * torch.rand(size = (3, pixel_num)) + normal_min

        else:
            normal = self.load_data.load_normal(i)
            
        return normal
        
    def createHyp(self, pixel_num, random, i):
        """ Create Hyperspectral reflectance for each pixels

            bring hyperspectral reflectance 
        
        """
        if random == True:
            min = 0.
            max = 1.29999           

            hyp = (max - min) * torch.rand(size=(pixel_num, self.wvls_n)) + min
            
        else:
            hyp = self.load_data.load_hyp_img(i)
            hyp = hyp.reshape(self.cam_H * self.cam_W, self.wvls_n)
        return hyp
    
    def createCamcoord(self, pixel_num, random):
       
        c, r = torch.meshgrid(torch.linspace(0,self.cam_H-1,self.cam_H), torch.linspace(0,self.cam_W-1,self.cam_W), indexing='ij') # 행렬 indexing
        c, r = c.reshape(self.cam_H*self.cam_W), r.reshape(self.cam_H*self.cam_W)
        ones = torch.ones_like(r)
        
        rc1_c = torch.stack((r,c,ones), dim = 0).permute(1,0)

        if random == True:
            rand_int = torch.randint(low = 0, high = self.cam_H*self.cam_W, size=(pixel_num,))
            return rc1_c[rand_int]

        else:
            return rc1_c
        
if __name__ == "__main__":
    
    arg = Argument()

    cam_coord = createData('coord', arg.cam_W*arg.cam_W).create()
    print('end')