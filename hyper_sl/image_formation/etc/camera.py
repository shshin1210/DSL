import torch
from scipy import interpolate
import numpy as np
import os

# import sys
# sys.path.append('/home/shshin/Scalable-Hyperspectral-3D-Imaging')
# from utils import constants as C

class Camera():
    def __init__(self, arg):
        
        ### camera intrinsic paramters
        self.N_cam = arg.cam_H
        self.focal_length_cam = arg.focal_length*1e-3
        self.sensor_width_cam = arg.sensor_width *1e-3
        self.cam_pitch = self.sensor_width_cam / self.N_cam
        
        self.crf_dir = arg.camera_response
        
        ### Response function parameters
        # wavelengths
        # self.wvls = arg.wvls
        # self.wvls_N = len(self.wvls)

        # # wavelengths for interpolation
        # self.wvls_int = torch.linspace(400*1e-9, 750*1e-9, 15)
        
        # CRF camera interpolation points 
        # self.CRF_cam_points = torch.tensor([[0.0605, 0.0704, 0.2429], # 400nm
        #                                     [0.0309, 0.0506, 0.4154], # 425nm
        #                                     [0.0166, 0.0526, 0.5269], # 450nm 
        #                                     [0.0167, 0.1851, 0.5237], # 475nm 
        #                                     [0.0167, 0.5859, 0.3684], # 500nm 
        #                                     [0.0429, 0.6513,0.1623], # 525nm 
        #                                     [0.0250, 0.6301, 0.0642], # 550nm 
        #                                     [0.0708, 0.5533, 0.0218], # 575nm 
        #                                     [0.6073,0.3882,0.0169], # 600nm 
        #                                     [0.6058,0.1740,0.0137], # 625nm 
        #                                     [0.5453,0.1004,0.0268], # 650nm 
        #                                     [0.4145,0.0858,0.0351], # 675nm 
        #                                     [0.0613,0.0237,0.0073], # 700nm 
        #                                     [0.0021,0.0021,0.0021], # 725nm
        #                                     [0.0009,0.0009,0.0009], # 750nm
        #                                     ]) 
    
    #### Unprojection
    def unprojection(self, depth):
        """ Unproject camera sensor coord plane to world coord

            input : depth
            return : world coordinate X,Y,Z
            
        """
        c, r = torch.meshgrid(torch.linspace(0,self.N_cam-1,self.N_cam), torch.linspace(0,self.N_cam-1,self.N_cam), indexing='ij') # 행렬 indexing
        
        x_c, y_c = (r-self.N_cam/2)*self.cam_pitch, (c-self.N_cam/2)*self.cam_pitch
        # x_c, y_c = x_c.reshape(self.N_cam*self.N_cam), y_c.reshape(self.N_cam*self.N_cam)
        # z_c = torch.zeros_like(x_c)
        # z_c[:] = -self.focal_length_cam
        
        # x_c, y_c= cam_coord[...,0], cam_coord[...,1]
        
        X,Y,Z = -x_c/self.focal_length_cam*depth, -y_c/self.focal_length_cam*depth, -depth
        
        return X,Y,Z
        
    #### Response function of Camera
    # def cubic_interpolation(self, x_new, x_points, y_points, n):
    #     """ interpolation function
        
    #     """
    #     tck = interpolate.splrep(x_points, y_points, k=n)   # Estimate the polynomial of nth degree by using x_points and y_points
    #     y_new = interpolate.splev(x_new, tck)
        
    #     return y_new
    
    # def compute_CRF(self):
    #     """ Compute the response function of Camera

    #         return : CRF
    #     """
    #     R_y_qe = self.CRF_cam_points[:,0] # R
    #     R_y_new = self.cubic_interpolation(self.wvls, self.wvls_int, R_y_qe , 2)

    #     G_y_qe = self.CRF_cam_points[:,1] # G
    #     G_y_new = self.cubic_interpolation(self.wvls, self.wvls_int, G_y_qe , 2)

    #     B_y_qe = self.CRF_cam_points[:,2] # B
    #     B_y_new = self.cubic_interpolation(self.wvls, self.wvls_int, B_y_qe , 2)
        
    #     CRF_cam = torch.zeros((self.wvls_N, 3))
        
    #     R_y_new = torch.tensor(R_y_new)
    #     G_y_new = torch.tensor(G_y_new)
    #     B_y_new = torch.tensor(B_y_new)

    #     CRF_cam[:,0] = R_y_new
    #     CRF_cam[:,1] = G_y_new
    #     CRF_cam[:,2] = B_y_new

    #     return CRF_cam
    
    def get_CRF(self):
        # CRF = self.compute_CRF()
        CRF = np.load(os.path.join(self.crf_dir, 'CRF_cam.npy'))
                
        return CRF
        
        