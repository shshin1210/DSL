import numpy as np
import torch
import cv2, os, sys

sys.path.append('/home/shshin/Scalable-Hyperspectral-3D-Imaging')

import glob
from hyper_sl.utils import open_exr
from hyper_sl.utils import crop

# import openEXR

class load_data():
    def __init__(self, arg):
        # path
        self.output_dir = arg.output_dir
        self.illum_path = arg.illum_dir
        self.img_hyp_text_path = arg.img_hyp_texture_dir
        self.openEXR = open_exr.openExr(arg)
        self.crop = crop
        
        # cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.sensor_width = (arg.sensor_width)*1e-3 # to
        self.focal_length = (arg.focal_length)*1e-3 # to meter
        
    def load_depth(self, i):
        """
        Bring distance numpy array and change it to depth tensor type

        return : depth (tensor)

        """
        dist = np.load(os.path.join(self.output_dir, "scene_%04d_Depth.npy" %(i))).astype(np.float32)
        dist = self.crop.crop(dist)
        
        dist = dist[...,0]
        x, y = torch.meshgrid(torch.linspace(-self.sensor_width/2, self.sensor_width/2, self.cam_H), torch.linspace(-self.sensor_width/2, self.sensor_width/2, self.cam_W))
        depth = self.focal_length *dist / torch.sqrt(x**2 + y**2 + self.focal_length**2)
        
        # depth to meter
        depth_to_meter = 10 # 1 - 1.4m
        depth *= depth_to_meter

        return depth
        
    def load_occ(self, i):
        """
        bring occlusion map

        input i-th scene occlusion map

        return occlusion map in tensor type

        """
        occlusion = np.load(os.path.join(self.output_dir, "scene_%04d_Occlusion.npy" %(i))).astype(np.float32)
        occlusion = self.crop.crop(occlusion)
        occlusion = torch.tensor(occlusion)
        occlusion = occlusion.reshape(self.cam_H*self.cam_W,3)

        # threshold
        mask_0 = (occlusion[:,:] <= 0.8)  #  0.5보다 작은 숫자들 즉, true 인 곳에 0을 넣기
        mask_1 = (occlusion[:,:] > 0.8)

        occ = np.ma.array(occlusion, mask=mask_0)
        occ = occ.filled(fill_value=0.0)

        occ = np.ma.array(occ, mask=mask_1)
        occ = occ.filled(fill_value=1.0)

        occ = occ.reshape(self.cam_H, self.cam_W,3)
        occ = occ[...,0]
        occ = torch.tensor(occ)

        return occ
    
    def load_illum(self, n):
        """
        bring illumination patterns

        input : illumination file name
            n : n-th illum pattern

        return illumination torch.tensor
        """
        
        files = sorted(glob.glob(os.path.join(self.illum_path,'*.png')))
        fn = files[n]
        
        illum = cv2.imread(fn).astype(np.float32)
        illum = illum / 255.
        illum = torch.tensor(illum)
        
        return illum
    
    def load_hyp_img(self, i):
        """"
        bring hyperspectral reflectance array
        
        """
        img_list = sorted(os.listdir(self.img_hyp_text_path))
        img_choice = img_list[i]
        
        hyp_img = np.load(os.path.join(self.img_hyp_text_path, img_choice)).astype(np.float32)
        hyp_img = cv2.resize(hyp_img, (1024, 768))
        hyp_img = hyp_img[...,2:27]
        hyp_img = self.crop.crop(hyp_img)

        hyp_img = torch.tensor(hyp_img)
        
        return hyp_img
    
    def load_normal(self, i):
        """
        bring normal array
        """
        # normal = self.openEXR.read_exr_as_np(i, "Normal").astype(np.float32)
        normal = np.load(os.path.join(self.output_dir, "scene_%04d_Normal.npy" %(i))).astype(np.float32)
        normal = self.crop.crop(normal)

        normal = torch.tensor(normal)
        normal = normal.reshape(self.cam_H*self.cam_W,3).transpose(1,0)

        norm = torch.norm(normal, dim = 0)
        normal_unit = normal/norm
        
        return normal_unit
    
    
if __name__ == "__main__":
    from ArgParser import Argument
    argument = Argument()
    arg = argument.parse()
    
    import matplotlib.pyplot as plt
    
    normal = load_data(arg).load_normal(0)
    # plt.imshow(depth), plt.colorbar()
    # plt.imsave('./output.png', depth)
    print('end')