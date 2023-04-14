import numpy as np
import torch, os
import scipy.io as sci

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils.load_data import load_data

class createData():
    def __init__(self, arg, data_type, pixel_num, random = True):

        # dir
        self.mat_dir = arg.img_hyp_texture_dir
        self.output_dir = arg.output_dir
        
        # arguments
        self.data_type = data_type
        self.pixel_num = pixel_num
        self.random = random
        self.arg = arg
        
        # params
        self.cam_res = arg.cam_W 
        self.cam_focal_length = arg.focal_length * 1e-3
        self.cam_sensor_width = arg.sensor_width *1e-3
        
        # class
        self.load_data = load_data(arg)
        
        
    def create(self):
        if self.data_type == "depth":
            depth = self.createDepth(self.pixel_num)
            return depth
        
        elif self.data_type == "hyp":
            hyp = self.createHyp(self.arg, self.pixel_num)
            return hyp
        
        elif self.data_type == "coord":
            cam_coord = self.createCamcoord(self.arg, self.pixel_num, self.random)
            return cam_coord
        
    def createDepth(self, pixel_num):
        """ Create Depth for each pixels

            pixel_num : # of pixels to create random depth data
            
            returns : torch.tensor ( # pixel, )
        """
        # meter 단위 depth
        depth_min = 1.
        depth_max = 1.4
        
        depth = (depth_max - depth_min)* torch.rand((pixel_num)) + depth_min
        # depth = depth.unsqueeze(dim=1)
        

    def createHyp(self, arg, pixel_num, random):
        """ Create Hyperspectral reflectance for each pixels

            bring hyperspectral reflectance 
        
        """
        if random:
            hyp = np.random.rand(pixel_num,arg.wvl_num).astype(np.float32)
        else:
            rand_int = torch.randint(low = 0, high = self.cam_res*self.cam_res, size=(pixel_num,))
            
            mat_file = os.path.join(self.mat_dir, f'T01array.mat')
            mat = sci.loadmat(mat_file)['cube'].reshape(self.cam_res*self.cam_res, -1) # 640*640, 77
            
            hyp = mat[rand_int] # pixel_num, 77
        
        return hyp
    
    def createCamcoord(self, arg, pixel_num, random):
        H, W = self.cam_res, self.cam_res
        
        c, r = torch.meshgrid(torch.linspace(0,H-1,H), torch.linspace(0,W-1,W), indexing='ij') # 행렬 indexing
        
        cam_pitch = self.cam_sensor_width/self.cam_res
        # cam coord의 x 좌표, y 좌표 
        x_c, y_c = (r-H/2)*cam_pitch, (c-W/2)*cam_pitch
        x_c, y_c = x_c.reshape(arg.cam_W*arg.cam_W), y_c.reshape(arg.cam_W*arg.cam_W)
        
        z_c = torch.zeros_like(x_c)
        z_c[:] = self.cam_focal_length
        
        xyz = torch.vstack((x_c, y_c, z_c)).permute(1,0) # pixel , 3
        
        if random == True:
            rand_int = torch.randint(low = 0, high = self.cam_res*self.cam_res, size=(pixel_num,))
            return xyz[rand_int]

        else:
            return xyz[:pixel_num]
        
if __name__ == "__main__":
    
    arg = Argument()

    cam_coord = createData('coord', arg.cam_W*arg.cam_W).create()
    print('end')