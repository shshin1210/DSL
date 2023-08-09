import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import calibrated_params
import numpy as np
import point_process
import matplotlib.pyplot as plt


class Define3dPoints():
    def __init__(self, arg, date, position):
        
        # arguments
        self.arg = arg
        self.date = date
        self.position = position
        
        # directory 
        self.main_dir = "./calibration/dg_calibration_method2/2023%s_data"%self.date
        self.data_dir = os.path.join(self.main_dir, self.position)
        self.point_dir = self.data_dir + '_points' # detected points dir
        self.points_3d_dir = os.path.join(self.main_dir , "spectralon_depth_%s_%s.npy"%(self.date,self.position)) # spectralon 3d points
        self.pattern_npy_dir = "./calibration/dg_calibration_method2/grid_npy" # pattern npy 
        
    def get_3d_points(self):
        points_3d = np.load(self.points_3d_dir)
        
        return points_3d

    def get_proj_px(self):
        proj_px = np.load(self.pattern_npy_dir)
        
        return proj_px

    def get_detected_pts(self, wvls, i, proj_px):
        detected_pts_dir = self.point_dir + '/pattern_%04d'%i
        detected_pts = point_process.point_process(self.arg, self.data_dir, detected_pts_dir, wvls, i, proj_px, self.position)
        detected_pts = (np.round(detected_pts)).astype(np.int32)
        
        return detected_pts

    def visualization(self, world_3d_pts):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        
        ax.scatter(world_3d_pts[:,:,:,0].flatten(), world_3d_pts[:,:,:,1].flatten(), world_3d_pts[:,:,:,2].flatten())

        ax.set_xlim([-0.15,0.15])
        ax.set_ylim([-0.1,0.1])
        ax.set_zlim([-0.5,0.6])

        plt.show()
        
    def world3d_pts(self):

        wvls = np.arange(450, 660, 50)
        wvls_num = len(wvls)
        total_px = (self.arg.proj_H//10)*(self.arg.proj_W//10) 

        # 3d points
        points_3d = self.get_3d_points()
        
        # New arrays : m, wvl, # px(=1), 2
        world_3d_pts = np.zeros(shape=(self.arg.m_num, wvls_num, total_px, 3))
        world_3d_pts_reshape = world_3d_pts.reshape(-1, total_px, 3) # m * wvl, # px, 3
        proj_pts = np.zeros(shape=(total_px, 2)) # projector sensor plane pxs : # px, 2
        
        for i in range(len(os.listdir(self.pattern_npy_dir))):
            # projector pixel points
            proj_px = self.get_proj_px()
            proj_pts[i] = proj_px
            
            # detected pts
            detected_pts = self.get_detected_pts(self.arg, wvls, i, proj_px) # m, wvl, 2
            detected_pts_reshape = detected_pts.reshape(-1, 2) # (x, y 순)
                
            world_3d_pts_reshape[:,i,:] = points_3d[detected_pts_reshape[:,1], detected_pts_reshape[:,0]]

        return world_3d_pts_reshape, proj_pts
        
    
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    date = "0728"
    position = ""

    front_world_3d_pts_reshape, proj_pts = Define3dPoints(arg, date, "front").world3d_pts()
    mid_world_3d_pts_reshape, proj_pts = Define3dPoints(arg, date, "mid").world3d_pts()
    back_world_3d_pts_reshape, proj_pts = Define3dPoints(arg, date, "back").world3d_pts()
    
    np.save('./front_world_3d_pts.npy', front_world_3d_pts_reshape)
    np.save('./back_world_3d_pts.npy', back_world_3d_pts_reshape)
    np.save('./proj_pts.npy', proj_pts)
    
    print('end')