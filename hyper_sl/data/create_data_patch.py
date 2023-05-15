import torch, os, cv2

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils.load_data import load_data
from hyper_sl.utils import data_process
from scipy.interpolate import interp1d
import numpy as np


class createData():
    def __init__(self, arg, data_type, pixel_num, random = True, i = 0):
        
        # dir
        self.mat_dir = arg.img_hyp_texture_dir
        self.output_dir = arg.output_dir
        self.img_hyp_texture_dir = arg.img_hyp_texture_dir
        self.real_data_dir = arg.real_data_dir
        
        # arguments
        self.data_type = data_type
        self.pixel_num = pixel_num
        self.random = random
        self.i = i
        self.arg = arg
        
        # params
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.illum_num = arg.illum_num
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
        
        else:
            N3_arr, illum_data = self.createReal(self.i)
            return N3_arr, illum_data
        
    def createDepth(self, pixel_num, random, i):
        """ Create Depth for each pixels

            pixel_num : # of pixels to create random depth data
            
            returns : torch.tensor ( # pixel, )
        """
        if random == True:
            # depth 끼리 차이 1mm ~ -1mm
            min = 0.001
            max = -0.001
            
            half = self.arg.patch_pixel_num // 2
            
            # meter
            depth_min = 0.6
            depth_max = 1.
            
            depth = (depth_max - depth_min)* torch.rand((pixel_num//self.arg.patch_pixel_num)) + depth_min
            depth = depth.repeat(9,1)
            
            tmp = (max - min) * torch.rand(size = (half, pixel_num //self.arg.patch_pixel_num)) + min
            depth[:half] = depth[:half] + tmp
            depth[half+1:] = depth[half+1:] + tmp
            
            depth = depth.T.reshape(-1,1).squeeze()
                        
        else:
            depth = self.load_data.load_depth(i)
            map_scale = interp1d([depth.min(), depth.max()], [0.6, 1.])
            depth = torch.tensor(map_scale(depth).astype(np.float32))
            
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
            
            # depth 끼리 차이 1mm ~ -1mm
            min = 0.001
            max = -0.001
            
            half = self.arg.patch_pixel_num // 2
            tmp = (max - min) * torch.rand(size = (half, 3, pixel_num //self.arg.patch_pixel_num)) + min

            normal = torch.zeros(size=(3, pixel_num//self.arg.patch_pixel_num))
            
            # normal
            normal_min = -0.9999
            normal_max = 0.9999
            
            normal = (normal_max - normal_min) * torch.rand(size = (3, pixel_num//self.arg.patch_pixel_num)) + normal_min
            normal = normal.repeat(9,1,1)
            
            normal[:half] = normal[:half] + tmp
            normal[half+1:] = normal[half+1:] + tmp
            
            normal = normal.permute(1,2,0).reshape(3,-1)
            normal_norm = torch.norm(normal, dim = 0)
            normal = normal / normal_norm
            
        else:
            normal = self.load_data.load_normal(i)
            
        return normal
        
    def createHyp(self, pixel_num, random, i):
        """ Create Hyperspectral reflectance for each pixels

            bring hyperspectral reflectance 
        
        """
        if random == True:
            # 3x3 patch, neighborhood hyp reflectance
            min = 0.001
            max = -0.001
            
            half = self.arg.patch_pixel_num // 2
            
            tmp = (max - min) * torch.rand(size = (half, pixel_num //self.arg.patch_pixel_num, 25)) + min
            
            ran_idx = torch.randint(0, len(os.listdir(self.img_hyp_texture_dir)), (1,))
            hyp = self.load_data.load_hyp_img(ran_idx)
            
            hyp = hyp.reshape(self.cam_H * self.cam_W, self.wvls_n)
            pixel_idx = torch.randint(0, self.cam_H*self.cam_W, (pixel_num//self.arg.patch_pixel_num,))           
            hyp = hyp[pixel_idx]
            hyp = hyp.repeat(self.arg.patch_pixel_num,1,1)
            
            hyp[:half] = hyp[:half] + tmp
            hyp[half+1:] = hyp[half+1:] + tmp
            
            hyp = hyp.permute(1,0,2).reshape(-1,25)
  
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
            rand_int = torch.randint(low = self.cam_W +1 , high = self.cam_H*self.cam_W - self.cam_W -1, size=(pixel_num//9,))
            
            top_l, top_m, top_r = rand_int - self.cam_W -1, rand_int - self.cam_W, rand_int - self.cam_W +1
            bot_l, bot_m, bot_r = rand_int + self.cam_W -1, rand_int + self.cam_W, rand_int + self.cam_W +1
            mid_l, mid_r = rand_int -1, rand_int +1
            
            patch_idx = torch.stack((top_l,top_m, top_r,mid_l,rand_int, mid_r, bot_l, bot_m, bot_r), dim = 0).T
            patch_idx = patch_idx.reshape(-1,1).squeeze()
            
            return rc1_c[patch_idx]

        else:
            return rc1_c
        
    def createReal(self, i):
        scene_i_dir = os.path.join(self.real_data_dir, 'scene%04d'%i)
        scene_files = sorted(os.listdir(scene_i_dir))
        
        N3_arr = torch.zeros(size=(self.cam_H*self.cam_W, self.illum_num, 3))
        
        for idx, fn in enumerate(scene_files):
            # Camera Undistortion 넣기
            if "Thumbs" in fn :
                continue
            else:
                real_img = cv2.imread(os.path.join(scene_i_dir, fn))
                # real_img = cv2.imread(os.path.join(scene_i_dir, fn), -1) / 65535.
                # real_img = data_process.crop(real_img)
                # cv2.imwrite('%s_img.png'%(fn[:-4]), real_img*255.)
                
                real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB).reshape(self.cam_H*self.cam_W,-1)/255.
                real_img = torch.tensor(real_img.reshape(self.cam_H*self.cam_W,-1))
                
                N3_arr[:,idx-1] = real_img

        illum_data = torch.zeros(size=(self.cam_H*self.cam_W, self.illum_num, self.wvls_n))
        
        return N3_arr, illum_data
        
if __name__ == "__main__":
    
    arg = Argument()

    cam_coord = createData('coord', arg.cam_W*arg.cam_W).create()
    print('end')