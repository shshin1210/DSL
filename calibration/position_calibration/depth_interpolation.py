import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
import matplotlib.pyplot as plt
from data_process import DataProcess
from scipy import interpolate
from scipy.io import loadmat, savemat
from tqdm import tqdm


class DepthInterpolation():
    def __init__(self, arg, date, front_peak_illum_idx, mid_peak_illum_idx, back_peak_illum_idx):
        
        # arguments
        self.date = date
        self.depth = arg.depth_list
        self.positions = ["front", "mid", "back"]
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        
        self.m_list = arg.m_list
        self.wvl_list = np.array([430, 660])
        self.depth_start = 600
        self.depth_end = 900     
        self.depth_arange = np.arange(self.depth_start, self.depth_end + 1, 1)
        self.sample_pts = np.array([[10 + i*215, 50 + j*106] for j in range(5) for i in range(5)])
        
        # delete points
        self.delete_pts = np.array([2 + 5*i - 1*i for i in range(5)])
        for i in range(len(self.delete_pts)):
            self.sample_pts = np.delete(self.sample_pts, self.delete_pts[i], axis = 0)

        self.sample_pts_flatt = np.array([[self.sample_pts[i,0]+self.sample_pts[i,1]*self.cam_W] for i in range(self.sample_pts.shape[0])]).squeeze()
        
        # dir
        self.data_dir = "./calibration/position_calibration/2023%s/%s_depth/spectralon"
        self.dat_dir = "./dataset/image_formation/dat/method3"
        
        # peak illumination index informations
        self.front_peak_illum_idx = front_peak_illum_idx # 3(m order), wvl(2), HxW
        self.mid_peak_illum_idx = mid_peak_illum_idx
        self.back_peak_illum_idx = back_peak_illum_idx
        
        # ppg graph
        self.max_data_back = np.load('./calibration/position_calibration/2023%s/npy_data/max_data_%s.npy'%(date, "back"))
        self.max_data_mid = np.load('./calibration/position_calibration/2023%s/npy_data/max_data_%s.npy'%(date, "mid"))
        self.max_data_front = np.load('./calibration/position_calibration/2023%s/npy_data/max_data_%s.npy'%(date, "front"))

    def depth_interpolation(self):
        """
            depth interpolation from 600mm to 900mm at 1mm interval
        
        """
        all_position_peak_illum_idx = np.stack((self.front_peak_illum_idx, self.mid_peak_illum_idx, self.back_peak_illum_idx), axis = 0)
        depth = np.array([self.get_depth(position) for position in self.positions]) * 1e+3
        
        depth_peak_illum_idx = np.zeros(shape=(len(self.depth_arange), len(self.m_list), len(self.wvl_list), len(self.sample_pts_flatt)))
        
        for m in range(len(self.m_list)):
            for w in range(len(self.wvl_list)):   
                for idx, i in enumerate(self.sample_pts_flatt):
                    # previous depth range and interpolated depth range
                    depth_range = np.round((np.array([depth[p,i] for p in range(len(self.positions))]))).astype(np.int32)
                    new_depth_range = np.arange(depth_range[0], depth_range[2] + 1, 1) 
                    
                    # exceptions
                    if (depth_range[0] > depth_range[1]) or (depth_range[1] > depth_range[2]) or (depth_range[0] > depth_range[2]) or (depth_range[0] > self.depth_start) or (depth_range[2] < self.depth_end):
                        depth_range = np.array([self.depth_start, self.depth_start + 100, self.depth_end])
                        new_depth_range = np.arange(self.depth_start, self.depth_end + 1, 1)
                        
                    # only bring 600mm to 900mm depths
                    idx_start, idx_end = np.where(new_depth_range == self.depth_start)[0][0], np.where(new_depth_range == self.depth_end)[0][0]
                    interp_depth = self.cubic_interpolation(new_depth_range, depth_range, all_position_peak_illum_idx[:,m,w,i].reshape(3, -1), 2)[idx_start:idx_end+1]
                    
                    # save depth
                    depth_peak_illum_idx[:, m, w, idx] = interp_depth
                    
        return depth_peak_illum_idx
    
    # bring depth values
    def get_depth(self, position):
        """
            bring depth for front/mid/back spectralon 
        """
        depth_dir = os.path.join(self.data_dir%(self.date, position), "2023%s_spectralon_%s.npy"%(self.date, position))
        depth = np.load(depth_dir)[:,:,2].reshape(self.cam_H * self.cam_W) # only get z(=depth) value

        return depth
    
    # interpolation
    def cubic_interpolation(self, x_new, x_points, y_points, n):
        tck = interpolate.splrep(x_points, y_points, k=n)   # Estimate the polynomial of nth degree by using x_points and y_points
        y_new = interpolate.splev(x_new, tck)
        
        return y_new
    
    # get depth dependent peak illumination indices
    def get_depth_peak_illum(self):
        """
            get depth dependent peak illumination index of zero orders and first orders
            
            shape :
            depth(600mm-900mm at 1mm interval), 2(m=-1 or 1 and 0), wvl(430nm, 660nm), sample pts
        """
        # depth_peak_illum_idx = self.depth_interpolation()
        # np.save('./depth_peak_illum_idx.npy', depth_peak_illum_idx)
        depth_peak_illum_idx = np.load('./depth_peak_illum_idx.npy')
        
                                                    # depth, m order (-1 or 1 and zero), wvl(430, 660), pts
        depth_peak_illum_idx_final = np.zeros(shape=(len(self.depth_arange), 2, len(self.wvl_list), self.sample_pts.shape[0])) 
        
        # masking out to get +1 or -1
        
        mask_pfirst = depth_peak_illum_idx[:,1,0] <= 200
        depth_peak_illum_idx_final[:,1,0][mask_pfirst] = depth_peak_illum_idx[:,2,0][mask_pfirst]
        depth_peak_illum_idx_final[:,1,1][mask_pfirst] = depth_peak_illum_idx[:,2,1][mask_pfirst]

        mask_mfirst = depth_peak_illum_idx[:,1,0] > 200
        depth_peak_illum_idx_final[:,1,0][mask_mfirst] = depth_peak_illum_idx[:,0,0][mask_mfirst]
        depth_peak_illum_idx_final[:,1,1][mask_mfirst] = depth_peak_illum_idx[:,0,1][mask_mfirst]

        # input zero order
        depth_peak_illum_idx_final[:,0] = depth_peak_illum_idx[:,1].mean(axis = 1)[:,np.newaxis,:]
        
        return depth_peak_illum_idx_final
        
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    date = "0922"
    
    # front_peak_illum_idx = DataProcess(arg, date, "front").get_first_idx()
    # mid_peak_illum_idx = DataProcess(arg, date, "mid").get_first_idx()
    # back_peak_illum_idx = DataProcess(arg, date, "back").get_first_idx()
    
    front_peak_illum_idx = np.load("./calibration/position_calibration/2023%s/npy_data/peak_illum_idx_front.npy"%date)
    mid_peak_illum_idx = np.load("./calibration/position_calibration/2023%s/npy_data/peak_illum_idx_mid.npy"%date)
    back_peak_illum_idx = np.load("./calibration/position_calibration/2023%s/npy_data/peak_illum_idx_back.npy"%date)
    
    DepthInterpolation(arg, date, front_peak_illum_idx, mid_peak_illum_idx, back_peak_illum_idx).get_depth_peak_illum()