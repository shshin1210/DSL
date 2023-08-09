import cv2, os, sys
import numpy as np

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from scipy.io import loadmat, savemat
from tqdm import tqdm

from define_3dpoints import Define3dPoints
from define_3dlines import Define3dLines
import file_process
from hyper_sl.utils.ArgParser import Argument

class CreateData():
    def __init__(self, arg, bool, date):
        
        # arguments
        self.arg = arg
        self.bool = bool
        self.date = date
        self.wvls = np.arange(450, 660, 50)
        self.pts_num = (arg.proj_H // 10) * (arg.proj_W // 10)
        self.m_list = arg.m_list
        self.depth = np.arange(0.6, 0.9, 0.01) # 10mm 간격
        # self.depth = np.arange(0.6, 0.9, 0.001) # 1mm 간격
        
        # classes
        self.processing_file = file_process.file_process()

        # directory
        self.dat_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/dat/method2"
        
    def createDepthData(self, front_world_3d_pts, dir_vec, proj_pts):
        for z in self.depth:
            t = (z - front_world_3d_pts[...,2]) / dir_vec[...,2]
            pts = front_world_3d_pts + dir_vec * t[:,:,:,np.newaxis]
            
            for m in tqdm(range(arg.m_num)):
                for w in tqdm(range(len(self.wvls)), leave = False):
                    
                    pts_x = pts[m, w, :, 0]
                    pts_y = pts[m, w, :, 1]
                    pts_z = pts[m, w, :, 2]
                    
                    gt_x = proj_pts[:,0]
                    gt_y = proj_pts[:,1]
                    
                    savemat(os.path.join(self.dat_path, 'dispersion_coordinates_m%d_wvl%d_depth%02dcm.mat'%(self.m_list[m], self.wvls[w], z *100)), {'x': pts_x, 'y': pts_y, 'z': pts_z, 'xo': gt_x, 'yo': gt_y})
                    # savemat(os.path.join(dat_path, 'dispersion_coordinates_m%d_wvl%d_depth%03dmm.mat'%(m_list[m], wvls[w], z *1000)), {'x': pts_x, 'y': pts_y, 'z': pts_z, 'xo': gt_x, 'yo': gt_y})

    def createData(self):
        # file processing : cropping, order datas in pattern and wavelengths
        self.processing_file(self.arg, self.bool, self.date, "front") # front spectralon
        self.processing_file(self.arg, self.bool, self.date, "mid") # mid spectralon
        self.processing_file(self.arg, self.bool, self.date, "back") # back spectralon
        
        # find 3d points of front & back spectralon
        front_world_3d_pts, proj_pts = Define3dPoints(self.arg, self.date, "front").world3d_pts()
        mid_world_3d_pts, proj_pts = Define3dPoints(self.arg, self.date, "mid").world3d_pts()
        back_world_3d_pts, proj_pts = Define3dPoints(self.arg, self.date, "back").world3d_pts()
        
        # save 3d points
        np.save('./front_world_3d_pts.npy', front_world_3d_pts)
        np.save('./mid_world_3d_pts.npy', mid_world_3d_pts)
        np.save('./back_world_3d_pts.npy', back_world_3d_pts)
        np.save('./proj_pts.npy', proj_pts)
        # bring saved 3d points
        front_world_3d_pts = np.load('./front_world_3d_pts.npy').reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        mid_world_3d_pts = np.load('./mid_world_3d_pts.npy').reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        back_world_3d_pts = np.load('./back_world_3d_pts.npy').reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        proj_pts = np.load('./proj_pts.npy')
        
        # 3d Line class
        defining_3dlines = Define3dLines(arg, front_world_3d_pts, mid_world_3d_pts, back_world_3d_pts)
        # visualization 3d points of specific order
        defining_3dlines.visualization(2)
        
        # define direction vector : m, wvl, #px, 3
        dir_vec = defining_3dlines.define3d_lines() # 형태로 구현하기
        
        # visualization of direction vector lines
        defining_3dlines.dir_visualization(dir_vec, 2)
    
        # save datas for each depths
        self.createDepthData(self, front_world_3d_pts.reshape(arg.m_num, len(self.wvls)* self.pts_num, 3), dir_vec.reshape(arg.m_num, len(self.wvls)* self.pts_num, 3) , proj_pts)
        
        
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    bool = False # True : undistort image / False : no undistortion to image
    date = "0728" # date of data
    
    # create mat data 
    CreateData(arg, bool, date).createData()