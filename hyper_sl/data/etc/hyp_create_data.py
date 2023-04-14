import numpy as np
import torch,sys

# sys.path.append('/home/shshin/Scalable-Hyperspectral-3D-Imaging')

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
        self.cam_res = arg.cam_W 
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
            cam_coord = self.createCamcoord(self.arg, self.pixel_num, self.random)
            return cam_coord

        elif self.data_type == "normal":
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
            depth = depth.reshape(self.cam_res, self.cam_res)
        else:
            depth = self.load_data.load_depth(i)
            
        return depth
    
    def createOcclusion(self, pixel_num, random, i):
        """ Create Depth for each pixels

            pixel_num : # of pixels to create random depth data
            
            returns : torch.tensor ( # pixel, )
        """
        if random == True:
            # no occlusions in random dataset
            occ = torch.ones(size = (pixel_num, ))
            occ = occ.reshape(self.cam_res, self.cam_res)
            
        else:
            occ = self.load_data.load_occ(i)
            
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
            hyp = (max - min) * torch.rand(size=(640,640,29)) + min
 
        else:
            
            hyp = self.load_data.load_hyp_img(i)
        return hyp
    
    def createRGB(self, pixel_num, random, i):
        """ Create Hyperspectral reflectance for each pixels

            bring hyperspectral reflectance 
        
        """
        if random == True:
            rgb = np.random.rand(pixel_num, 3).astype(np.float32) # (pixel, 29)
            rgb = rgb.reshape(self.cam_res, self.cam_res, 3)
        else:
            
            rgb = self.load_data.load_albedo(i)
        
        return rgb
    
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
            return xyz
        
if __name__ == "__main__":
    import torch,sys

    # sys.path.append('/home/shshin/Scalable-Hyperspectral-3D-Imaging')

    from hyper_sl.utils.ArgParser import Argument
    argument = Argument()
    arg = argument.parse()
    

    for i in range(10):
        hyp = createData('hyp', arg.cam_W*arg.cam_W, False, i).create()
    print('end')