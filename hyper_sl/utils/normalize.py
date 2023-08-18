import torch

from hyper_sl.image_formation_method2.projector import Projector

class Normalize():
    def __init__(self, arg):
        # argument
        self.arg = arg
        
        # class
        self.proj = Projector(arg, device= arg.device)
        
        # proj
        self.proj_focal_length = self.proj.focal_length_proj()        
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.intrinsic_proj_real = self.proj.intrinsic_proj_real()

    def proj_sensor_plane(self):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """
        #  proj sensor
        xs = torch.linspace(0,self.proj_H-1, self.proj_H)
        ys = torch.linspace(0,self.proj_W-1, self.proj_W)
        r, c = torch.meshgrid(xs, ys, indexing='ij')
        
        c, r = c.flatten(), r.flatten()
        ones = torch.ones_like(c)
        cr1 = torch.stack((c,r,ones), dim = 0)
        xyz = (torch.linalg.inv(self.intrinsic_proj_real)@(cr1*self.proj_focal_length))
        
        return xyz
    
    def normalization(self,data):   
        """
            normalize projector coord xy
        """
        
        xyz = self.proj_sensor_plane()
        x_min, x_max =  xyz[0].min(), xyz[0].max()
        y_min, y_max =  xyz[1].min(), xyz[1].max()

        x_gt_min, x_gt_max = torch.tensor([x_min - 1e-4], device= self.arg.device), torch.tensor([x_max + 1e-4], device= self.arg.device)
        y_gt_min, y_gt_max = torch.tensor([y_min - 1e-4], device= self.arg.device), torch.tensor([y_max + 1e-4], device= self.arg.device)
        
        x_gt_minmax = (data[...,0] - x_gt_min.unsqueeze(dim = 1)) / (x_gt_max.unsqueeze(dim = 1) - x_gt_min.unsqueeze(dim = 1)) 
        y_gt_minmax = (data[...,1] - y_gt_min.unsqueeze(dim = 1)) / (y_gt_max.unsqueeze(dim = 1) - y_gt_min.unsqueeze(dim = 1))
        
        xy_proj_mm = torch.stack((x_gt_minmax, y_gt_minmax), dim = 2)
            
        return xy_proj_mm

    def un_normalization(self,data):
        """
            unnormalize projector coord xy
        """
        
        xyz = self.proj_sensor_plane()
        x_min, x_max =  xyz[0].min(), xyz[0].max()
        y_min, y_max =  xyz[1].min(), xyz[1].max()
        
        data_unnorm_x = data[...,0] * (x_max - x_min) + x_min
        data_unnorm_y = data[...,1] * (y_max - y_min) + y_min

        data_unnorm = torch.stack((data_unnorm_x, data_unnorm_y), dim = 2)
        
        return data_unnorm    

    def N3_normalize(self, N3_arr, illum_num):
        """
            normalization of N3_arr
        """
        
        N3_arr_r = N3_arr.reshape(-1, illum_num*3)
        N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
        N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
        N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)

        return N3_arr_normalized
