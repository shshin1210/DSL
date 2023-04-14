import numpy as np
import torch
import cv2, os

import sys
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset')
from data import generate_hyp


class Pattern_formation():
    def __init__(self, arg):
        # path
        self.output_dir = arg.output_dir
        self.illum_path = arg.illum_path
        self.param_path = arg.param_path

        # cam
        self.cam_resolution = arg.cam_resolution
        self.sensor_width = (arg.sensor_width)*1e-3
        self.focal_length = (arg.focal_length)*1e-3

        # proj
        self.sensor_diag_proj = arg.sensor_diag_proj
        self.focal_length_proj = arg.focal_length_proj
        self.proj_resolution = arg.proj_resolution

    def patt_form(self, i):
        # bring depth
        depth = self.depth(i)

        # unproject to world coord from cam sensor
        X,Y,Z = self.unprojection(depth= depth)

        # projection world coord to proj coord
        rc_proj,_ = self.projection(X,Y,Z)

        # illumination
        # ===================================== illumination file name 나중에 결정 ======================================
        illum = self.bring_illum("pattern_12")
        illum_img = illum[rc_proj[:,:,1].long(), rc_proj[:,:,0].long()]

        # multiply with occlusion
        occ = self.bring_occ(i)
        illum_w_occ = illum_img * occ

        return illum_w_occ

    def depth(self, i):
        """
        Bring distance numpy array and change it to depth tensor type

        return : depth (tensor)

        """
        dist = generate_hyp.Hyper_Generator.read_exr_as_np(self, i, "Depth").astype(np.float32)
        dist = dist[...,0]
        N = self.cam_resolution
        x, y = np.meshgrid(np.linspace(-self.sensor_width/2, self.sensor_width/2, N), np.linspace(-self.sensor_width/2, self.sensor_width/2, N))
        depth = self.focal_length *dist / np.sqrt(x**2 + y**2 + self.focal_length**2)
        
        # depth to meter
        depth_to_meter = 10
        depth *= depth_to_meter
        depth = torch.tensor(depth)

        return depth

    def unprojection(self, depth):
        """
        cam sensor (r,c) to world coordinate

        """
        R,C = depth.shape
        xs = torch.linspace(0, R-1, steps = R)
        ys = torch.linspace(0, C-1, steps= C)

        c,r = torch.meshgrid(xs, ys, indexing='ij')

        cam_pitch = self.sensor_width/self.cam_resolution
        x_c, y_c = (r-R/2)*cam_pitch, (c-C/2)*cam_pitch
        z_c = torch.zeros_like(x_c)
        z_c[:] = -self.focal_length

        X,Y,Z = (-x_c/self.focal_length)*depth, (-y_c/self.focal_length)*depth, depth

        return X,Y,Z
    
    def projection(self, X,Y,Z):
        """
        project world coordinate to projector coordinate

        return projector sensor coordinate in tensor type

        """
        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        XYZ1 = torch.stack((X,Y,Z, torch.ones_like(X)), dim = 1).transpose(1,0)

        # bring parameters
        # proj_int = self.bring_param("proj_int")
        # cam_proj_tvec = self.bring_param("cam_proj_tvec")

        # projector sensor pitch, fov
        sensor_width_proj = np.sin(np.arctan2(720,1280)) * self.sensor_diag_proj
        proj_pitch = sensor_width_proj/self.proj_resolution
        proj_fov = np.rad2deg(np.arctan2(sensor_width_proj/2, self.focal_length_proj))

        # proj extrinsic matrix
        proj_extrinsic = self.proj_extrinsic()

        # world coord XYZ at proj view
        XYZ_proj = proj_extrinsic@XYZ1

        # world coord to proj sensor 
        rc_proj = (-self.focal_length_proj*XYZ_proj[:2,:]/XYZ_proj[2,:])/proj_pitch + self.proj_resolution/2
        R, C = self.cam_resolution, self.cam_resolution
        rc_proj = rc_proj.transpose(0,1).reshape(R,C,2)

        return rc_proj, XYZ_proj
    
    def bring_occ(self, i):
        """
        bring occlusion map

        input i-th scene occlusion map

        return occlusion map in tensor type

        """
        occlusion = generate_hyp.Hyper_Generator.read_exr_as_np(self, i, "Occlusion").astype(np.float32)
        occlusion = torch.tensor(occlusion)
        occlusion = occlusion.reshape(self.cam_resolution*self.cam_resolution,3)

        # threshold
        mask_0 = (occlusion[:,:] <= 0.8)  #  0.5보다 작은 숫자들 즉, true 인 곳에 0을 넣기
        mask_1 = (occlusion[:,:] > 0.8)

        occ = np.ma.array(occlusion, mask=mask_0)
        occ = occ.filled(fill_value=0.0)

        occ = np.ma.array(occ, mask=mask_1)
        occ = occ.filled(fill_value=1.0)

        occ = occ.reshape(self.cam_resolution,self.cam_resolution,3)
        occ = torch.tensor(occ)

        return occ


    def bring_illum(self, illum):
        """
        bring illumination patterns

        input : illumination file name

        return illumination torch.tensor
        """
        
        illum_file = illum + '.png'
        illum = cv2.imread(os.path.join(self.illum_path, illum_file)).astype(np.float32)
        illum = torch.tensor(illum)

        R, C = illum.shape[0], illum.shape[1]

        # change [255,255,255]  to [1,1,1]
        illum = illum.reshape(R*C, 3)

        mask_0 = (illum[:,:] < 150)
        mask_1 = (illum[:,:] >= 150)

        ill = np.ma.array(illum, mask = mask_0)
        ill = ill.filled(fill_value = 0.0)

        ill = np.ma.array(ill, mask= mask_1)
        ill = ill.filled(fill_value = 1.0)

        ill = ill.reshape(R,C, 3)
        ill = torch.tensor(ill)

        return ill
    
    def proj_extrinsic(self):
        """
        create projector's extrinsic matrix

        return projector's extrinsic matrix in tensor type

        """
        # proj extrinsic matrix
        proj_extrinsic = np.zeros((3,4))
        # no rotation
        proj_extrinsic[0,0] = 1 
        proj_extrinsic[1,1] = 1
        proj_extrinsic[2,2] = 1

        # translate + x 50e-3
        proj_extrinsic[0,3] = 50e-3 

        # change to tensor
        proj_extrinsic = torch.tensor(proj_extrinsic)

        return proj_extrinsic


    def bring_param(self, param):
        """
        bring intrinsic or extrinsic matrix / parameters
        
        return parameter

        """
        param_file = param + ".npy"
        parameter = torch.tensor(np.load(os.path.join(self.param_path, param_file)))
        
        return parameter


    def to_proj_plane(self, X,Y,Z):

        print('end')
