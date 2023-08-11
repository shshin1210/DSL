import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from scipy.io import loadmat, savemat
from tqdm import tqdm

from define_3dpoints import Define3dPoints
from define_3dlines import Define3dLines
from file_process import FileProcess
from hyper_sl.utils.ArgParser import Argument

class CreateData():
    def __init__(self, arg, bool, date):
        
        # arguments
        self.arg = arg
        self.bool = bool
        self.date = date
        self.wvls = np.arange(450, 660, 50)
        self.interpolated_wvls = np.arange(420, 670, 10)
        self.pts_num = (arg.proj_H // 10) * (arg.proj_W // 10)
        self.m_list = np.array([-1, 1])
        self.depth = np.arange(0.6, 0.9, 0.01) # 10mm 간격
        # self.depth = np.arange(0.6, 0.9, 0.001) # 1mm 간격
        self.num_param = 21
        
        # classes
        self.processing_file = FileProcess(self.arg, self.date)

        # directory
        self.dat_dir = "./dataset/image_formation/dat/method2"
        self.data_npy_dir = "./calibration/dg_calibration_method2/2023%s_data"%date
        self.param_dir = "./dataset/image_formation/dat/method2"
    
    def load_param(self):
        """
            bring parameters before interpolation
        """
        param_list = np.zeros(shape=(len(self.m_list), len(self.wvls), len(self.depth), 21, 2))

        for z in tqdm(range(len(self.depth))):
            for m in tqdm(range(len(self.m_list))):
                    for w in tqdm(range(len(self.wvls)), leave = False):
                        param_list[m, w, z] = loadmat(os.path.join(self.param_dir, 'param_dispersion_coordinates_m%d_wvl%d_depth%dcm.mat' %(self.m_list[m], self.wvls[w], self.depth[z]*100)))['p'].T
        
        return param_list
    
    def interpolate_data(self):
        """
            interpolate for other wavelength coefficients
            
            save coefficients for other wavelengths
        """
        
        param_list = self.load_param()
        interpolated_param_list = np.zeros(shape=(len(self.m_list), len(self.interpolated_wvls), len(self.depth), self.num_param, 2))
        
        # 저장해야할 interpolated parameter shape
        # m (2), wvl (25), depth (31), param (21), xy (2)
        
        # do interpolation
        for z in tqdm(range(len(self.depth))):
            for m in tqdm(range(len(self.m_list))):
                    for w in tqdm(range(len(self.wvls)), leave = False):
                        for i in tqdm(range(self.num_param)):
                            for xy in tqdm(range(2)):
                                params = param_list[m,:,z,i,xy]
                                interpolated_param_list[m,:,z,i,xy] = self.cubic_interpolation(self.interpolated_wvls, self.wvls, params, 2)
        
        # save interpolated polynomial function coefficients
        for z in tqdm(range(len(self.depth))):
            for m in tqdm(range(len(self.m_list))):
                    for w in tqdm(range(len(self.interpolated_wvls)), leave = False):            
                        savemat(os.path.join(self.dat_dir, 'interpolated/param_dispersion_coordinates_m%d_wvl%d_depth%02dcm.mat'%(self.m_list[m], self.interpolated_wvls[w], self.depth[z] *100)), {'p': interpolated_param_list[m,w,z]})

    def cubic_interpolation(self, x_new, x_points, y_points, n):
        tck = interpolate.splrep(x_points, y_points, k=n)   # Estimate the polynomial of nth degree by using x_points and y_points
        y_new = interpolate.splev(x_new, tck)
        return y_new

    def createDepthData(self, front_world_3d_pts, dir_vec, proj_pts):
        """
            create point datasets with different depths
            
            depth : 0.6 m ~ 0.9 m / 1cm interval or 1mm interval
            (change self.depth arange for interval difference!!)
            
        """
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
                    
                    savemat(os.path.join(self.dat_dir, 'dispersion_coordinates_m%d_wvl%d_depth%02dcm.mat'%(self.m_list[m], self.wvls[w], z *100)), {'x': pts_x, 'y': pts_y, 'z': pts_z, 'xo': gt_x, 'yo': gt_y})
                    # savemat(os.path.join(dat_dir, 'dispersion_coordinates_m%d_wvl%d_depth%03dmm.mat'%(m_list[m], wvls[w], z *1000)), {'x': pts_x, 'y': pts_y, 'z': pts_z, 'xo': gt_x, 'yo': gt_y})

    def process_file(self):
        # file processing : cropping, order datas in pattern and wavelengths
        self.processing_file.file_process(self.bool, "front") # front spectralon
        self.processing_file.file_process(self.bool, "mid") # mid spectralon
        self.processing_file.file_process(self.bool, "back") # back spectralon
    
    def find_3d_points(self):
        # find 3d points of front & back spectralon
        front_world_3d_pts, proj_pts = Define3dPoints(self.arg, self.date, "front").world3d_pts()
        mid_world_3d_pts, proj_pts = Define3dPoints(self.arg, self.date, "mid").world3d_pts()
        back_world_3d_pts, proj_pts = Define3dPoints(self.arg, self.date, "back").world3d_pts()
        
        return front_world_3d_pts, mid_world_3d_pts, back_world_3d_pts, proj_pts
    
    def dir_outlier(self, dir_vec):
        """
            delete second outliers of direction vector
            (ouliers caused by intensity mistakes for edge points)
            
        """
        
        dir_vec_reshape = dir_vec.reshape(-1, 3)
        
        for i in range((dir_vec_reshape.shape[0])):
            if dir_vec_reshape[i,0] > 0.1:
                dir_vec_reshape[i] = np.zeros(shape=(3,))
                
        return dir_vec
    
    def createData(self):
        # file processing : cropping, order datas in pattern and wavelengths
        self.process_file()
        
        # find 3d points of front & back spectralon
        front_world_3d_pts, mid_world_3d_pts, back_world_3d_pts, proj_pts = self.find_3d_points()
        
        # save 3d points
        np.save(os.path.join(self.data_npy_dir,'front_world_3d_pts.npy'), front_world_3d_pts)
        np.save(os.path.join(self.data_npy_dir,'mid_world_3d_pts.npy'), mid_world_3d_pts)
        np.save(os.path.join(self.data_npy_dir,'back_world_3d_pts.npy'), back_world_3d_pts)
        np.save(os.path.join(self.data_npy_dir,'proj_pts.npy'), proj_pts)
        
        # bring saved 3d points
        front_world_3d_pts = np.load(os.path.join(self.data_npy_dir,'front_world_3d_pts.npy')).reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        mid_world_3d_pts = np.load(os.path.join(self.data_npy_dir,'mid_world_3d_pts.npy')).reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        back_world_3d_pts = np.load(os.path.join(self.data_npy_dir,'back_world_3d_pts.npy')).reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        proj_pts = np.load(os.path.join(self.data_npy_dir,'proj_pts.npy'))
        
        # 3d Line class
        defining_3dlines = Define3dLines(arg, front_world_3d_pts, mid_world_3d_pts, back_world_3d_pts)
        # visualization 3d points of specific order
        defining_3dlines.visualization(2)
        
        # define direction vector : m, wvl, # px, 3
        dir_vec, start_pts = defining_3dlines.define3d_lines()
        # direction vector outlier
        dir_vec = self.dir_outlier(dir_vec)
        
        np.save(os.path.join(self.data_npy_dir,'dir_vec.npy'), dir_vec)
        np.save(os.path.join(self.data_npy_dir,'start_pts.npy'), start_pts)
        
        dir_vec = np.load(os.path.join(self.data_npy_dir,'dir_vec.npy')).reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        start_pts = np.load(os.path.join(self.data_npy_dir,'start_pts.npy')).reshape(arg.m_num, len(self.wvls), self.pts_num, 3)
        
        # visualization of direction vector lines and points
        defining_3dlines.dir_visualization(dir_vec, start_pts, 0, 0)
    
        # save datas for each depths
        self.createDepthData(start_pts, dir_vec, proj_pts)
        
        # save interpolated parameters
        self.interpolate_data()
        
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    bool = False # True : undistort image / False : no undistortion to image
    date = "0728" # date of data
    
    # create mat data 
    CreateData(arg, bool, date).createData()