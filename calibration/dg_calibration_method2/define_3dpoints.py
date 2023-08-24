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
        self.wvls = np.array([430, 450, 480, 500, 520, 550, 580, 600, 620, 650, 660])
        # self.total_px = (self.arg.proj_H//10)* (self.arg.proj_W//10)
        self.total_px = arg.total_px
        self.epoch = 1099
        self.cam_int, _ = calibrated_params.bring_params(arg.calibration_param_path, "cam")

        # directory 
        self.main_dir = "./calibration/dg_calibration_method2/2023%s_data"%self.date
        self.data_dir = os.path.join(self.main_dir, self.position)
        self.processed_dir = os.path.join(self.main_dir, "%s_processed"%self.position)
        self.point_dir = self.data_dir + '_points' # detected points dir
        self.points_3d_dir = os.path.join(self.main_dir , "spectralon_depth_%s_%s.npy"%(self.date,self.position)) # spectralon 3d points
        self.pattern_npy_dir = "./calibration/dg_calibration_method2/grid_npy" # pattern npy 
        
    def get_3d_points(self):
        """
            get gray code decoded 3d points
        """
        points_3d = np.load(self.points_3d_dir)
        
        return points_3d

    def get_proj_px(self, dir):
        """
            get (u, v) coords for projected projector sensor plane
        """
        proj_px = np.load(dir)
        
        return proj_px

    def get_detected_pts(self, i, proj_px):
        """
            get preprocessed detection points : m, wvl, # points, 3(xyz)
            
            returns detected points
            
        """
        detected_pts_dir = self.point_dir + '/pattern_%04d'%i
        processed_img_dir = self.processed_dir + '/pattern_%04d'%i
        detected_pts = point_process.PointProcess(self.arg, self.data_dir, detected_pts_dir, processed_img_dir, self.wvls, i, proj_px, self.position).point_process()
        undistort_pts = self.undistort(detected_pts=detected_pts)
        undistort_pts = (np.round(undistort_pts)).astype(np.int32) # m, wvl, 2

        return undistort_pts
    
    def undistort(self, detected_pts):
        undistort_pts = np.zeros(shape=(self.arg.m_num, len(self.wvls), 2))
        
        for w in range(len(self.wvls)):
            opt_param = np.load(os.path.join(self.main_dir, 'opt_param/%sparam_%06d.npy'%(self.position, self.epoch))) # wvls, 5

            undistort_detected_pts = cv2.undistortPoints(src = detected_pts[:,w].reshape(-1, 1, 2), cameraMatrix = self.cam_int, distCoeffs= opt_param[w])
            undistort_pts[:, w] = undistort_detected_pts.squeeze()
            
            ones = np.ones(shape=(3, len(self.wvls), 1))
            undistort_pts1 = np.concatenate((undistort_pts, ones), axis = 2)
            undistort_pts_uv1 = self.cam_int@undistort_pts1.reshape(-1, 3).transpose(1, 0)
            undistort_pts_uv = undistort_pts_uv1.transpose(1, 0).reshape(self.arg.m_num, len(self.wvls), 3)[...,:2]
        
        return undistort_pts_uv

    def visualization(self, world_3d_pts):
        """
            visualization of 3d points
        """
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        
        ax.scatter(world_3d_pts[:,:,:,0].flatten(), world_3d_pts[:,:,:,1].flatten(), world_3d_pts[:,:,:,2].flatten())

        ax.set_xlim([-0.15,0.15])
        ax.set_ylim([-0.1,0.1])
        ax.set_zlim([-0.5,0.6])

        plt.show()
        
    def world3d_pts(self):
        """
            Get world 3d points by detected points
            
        """
        # wvls = np.arange(450, 660, 50)
        wvls_num = len(self.wvls)
        
        # 3d points
        points_3d = self.get_3d_points()
        
        # New arrays : m, wvl, # px(=1), 2
        world_3d_pts = np.zeros(shape=(self.arg.m_num, wvls_num, self.total_px, 3))
        world_3d_pts_reshape = world_3d_pts.reshape(-1, self.total_px, 3) # m * wvl, # px, 3
        proj_pts = np.zeros(shape=(self.total_px, 2)) # projector sensor plane pxs : # px, 2
        
        for i in range(len(os.listdir(self.pattern_npy_dir))):
            # projector pixel points
            proj_px = self.get_proj_px(os.path.join(self.pattern_npy_dir,"pattern_%05d.npy"%i))
            proj_pts[i] = proj_px
            
            # detected pts
            detected_pts = self.get_detected_pts(i, proj_px) # m, wvl, 2
            detected_pts_reshape = detected_pts.reshape(-1, 2) # (x, y 순)

            world_3d_pts_reshape[:,i,:] = points_3d[detected_pts_reshape[:,1], detected_pts_reshape[:,0]]

            # (u, v) = (0, 0) 인 point에 대해서는 말도안되는 값을 넣어줘서 outlier 처리가 될 수 있게끔 처리
            for j in range(detected_pts_reshape.shape[0]):
                if (detected_pts_reshape[j] == 0.).all():
                    world_3d_pts_reshape[j,i,:] = np.array([-0.5,-0.5,-0.5])
                    
        return world_3d_pts_reshape, proj_pts
        
    
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    date = "0822"

    front_world_3d_pts_reshape, proj_pts = Define3dPoints(arg, date, "front").world3d_pts()
    mid_world_3d_pts_reshape, proj_pts = Define3dPoints(arg, date, "mid").world3d_pts()
    back_world_3d_pts_reshape, proj_pts = Define3dPoints(arg, date, "back").world3d_pts()
    
    np.save('./front_world_3d_pts.npy', front_world_3d_pts_reshape)
    np.save('./back_world_3d_pts.npy', back_world_3d_pts_reshape)
    np.save('./proj_pts.npy', proj_pts)
    
    print('end')