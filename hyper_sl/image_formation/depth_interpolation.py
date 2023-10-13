import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
import matplotlib.pyplot as plt
from data_process import DataProcess
from scipy import interpolate
from scipy.io import loadmat, savemat
from tqdm import tqdm
from scipy.optimize import curve_fit


"""
    def depth_interpolation : interpolate 5 depths to 301 depths with non-linear fitting

    def get_depth : bring depth values (mm) for each spectralon position
    
    det_depth_peak_illum(self):
        
            Picking the valid first orders between +1 and -1
            get depth dependent peak illumination index of zero orders and only one first order
            
            shape :
            depth(600mm-900mm at 1mm interval), 2(0 and -1/+1), wvl(430nm, 600nm - 660nm), sample pts
            
"""

class DepthInterpolation():
    def __init__(self, arg, date, front_peak_illum_idx, mid_peak_illum_idx, mid2_peak_illum_idx, mid3_peak_illum_idx, back_peak_illum_idx):
        
        # arguments
        self.date = date
        self.depth = arg.depth_list
        self.positions = ["front", "mid","mid2","mid3", "back"]
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        
        self.m_list = arg.m_list
        self.wvl_list = np.array([430, 600, 610, 620, 640, 650, 660])
        self.depth_start = 600
        self.depth_end = 900     
        self.depth_arange = np.arange(self.depth_start, self.depth_end + 1, 1)
        
        self.sample_pts = np.array([[10 + i*120, 50 + j*51] for j in range(10) for i in range(8)])
        # self.sample_pts = np.array([[10 + i*60, 50 + j*51] for j in range(10) for i in range(15)])
        self.sample_pts_flatt = np.array([[self.sample_pts[i,0]+self.sample_pts[i,1]*self.cam_W] for i in range(self.sample_pts.shape[0])]).squeeze()
        
        # dir
        self.to_depth_dir = "./dataset/image_formation/2023%s/%s_depth/spectralon"
        self.npy_dir = "./dataset/image_formation/2023%s/npy_data"%date

        # peak illumination index informations / 3(m order), wvl, HxW
        self.front_peak_illum_idx = front_peak_illum_idx
        self.mid_peak_illum_idx = mid_peak_illum_idx
        self.mid2_peak_illum_idx = mid2_peak_illum_idx
        self.mid3_peak_illum_idx = mid3_peak_illum_idx
        self.back_peak_illum_idx = back_peak_illum_idx

    def depth_interpolation(self):
        """
            depth interpolation from 600mm to 900mm at 1mm interval
        
        """
        all_position_peak_illum_idx = np.stack((self.front_peak_illum_idx, self.mid_peak_illum_idx, self.mid2_peak_illum_idx, self.mid3_peak_illum_idx, self.back_peak_illum_idx), axis = 0)
        depth = np.array([self.get_depth(position) for position in self.positions]) * 1e+3
        
        depth_peak_illum_idx = np.zeros(shape=(len(self.depth_arange), len(self.m_list), len(self.wvl_list), len(self.sample_pts_flatt)))
        # for visualization
        # vis_list = []
        # vis_range = []
        
        ## DEBUG HERE =====================================================================================================================================
        
        
        for m in range(len(self.m_list)):
                for w in range(len(self.wvl_list)): 
                    for idx, i in enumerate(self.sample_pts_flatt):  
                        depth_range = np.round(np.array([depth[p,i] for p in range(5)])).astype(np.int32)
                                        
                        # non linear fitting
                        if all_position_peak_illum_idx[:,m,w,i].mean() < 1 : 
                            all_position_peak_illum_idx[:,m,w,i] = np.array([0, 0, 0, 0, 0])
                            
                        new_depth_range = np.arange(depth_range[0], depth_range[-1] + 1, 1) 
                        idx_start, idx_end = np.where(new_depth_range == self.depth_start)[0][0], np.where(new_depth_range == self.depth_end)[0][0]
                        cnt_317 = np.count_nonzero(all_position_peak_illum_idx[:,m,w,i].reshape(len(depth_range)).flatten().astype(np.int16) == 317)
                        
                        if (1 < cnt_317 < 4):
                            polynom = np.interp(new_depth_range, depth_range, all_position_peak_illum_idx[:,m,w,i].reshape(len(depth_range)).flatten(), 6)
                            depth_peak_illum_idx[:, m, w, idx] = polynom[idx_start:idx_end+1]
                            
                        else: 
                            params, cov = curve_fit(self.fitting_function, depth_range, all_position_peak_illum_idx[:,m,w,i].reshape(len(depth_range)).flatten(), maxfev = 500000)
                            interp_depth = self.fitting_function(new_depth_range, *params)
                            final_depth = interp_depth[idx_start:idx_end+1]
                            depth_peak_illum_idx[:, m, w, idx] = final_depth 

                        # polynom = np.interp(new_depth_range, depth_range, all_position_peak_illum_idx[:,m,w,i].reshape(len(depth_range)).flatten(), 6)

                        # depth_peak_illum_idx[:, m, w, idx] = final_depth 
                        # depth_peak_illum_idx_linear[:, m, w, idx] = polynom[idx_start:idx_end+1]


        # for m in range(len(self.m_list)):
        #     for w in range(len(self.wvl_list)):   
        #         for idx, i in enumerate(self.sample_pts_flatt):

        #             # previous depth range and interpolated depth range
        #             depth_range = np.round(np.array([depth[p,i] for p in range(len(self.positions))])).astype(np.int32)
        #             new_depth_range = np.arange(depth_range[0], depth_range[-1] + 1, 1) 
                    
        #             # exceptions
        #             if (False in (np.sort(depth_range) == depth_range)) or (depth_range[0] > self.depth_start) or (depth_range[-1] < self.depth_end):
        #                 depth_range = np.array([self.depth_start, self.depth_start + 100, self.depth_end])
        #                 new_depth_range = np.arange(self.depth_start, self.depth_end + 1, 1)
                        
        #             # only bring 600mm to 900mm depths
        #             idx_start, idx_end = np.where(new_depth_range == self.depth_start)[0][0], np.where(new_depth_range == self.depth_end)[0][0]
        #             # non linear fitting
        #             # print(all_position_peak_illum_idx[:,m,w,i])
                    
        #             # non linear fitting
        #             if all_position_peak_illum_idx[:,m,w,i].mean() < 1 : 
        #                 all_position_peak_illum_idx[:,m,w,i] = np.array([0, 0, 0, 0, 0])
                        
        #             # params, cov = curve_fit(self.fitting_function, depth_range, all_position_peak_illum_idx[:,m,w,i].reshape(len(depth_range)).flatten(), maxfev = 500000000)
        #             # interp_depth = self.fitting_function(new_depth_range, *params)
        #             polynom = np.interp(new_depth_range, depth_range, all_position_peak_illum_idx[:,m,w,i].reshape(len(depth_range)).flatten(), 6)

        #             # print(interp_depth[::30])
                    
        #             # final_depth = interp_depth[idx_start:idx_end+1]
        #             final_depth = polynom[idx_start:idx_end+1]

        #             # save depth
        #             depth_peak_illum_idx[:, m, w, idx] = final_depth 
                    
        #             # for visualization
        #             # vis_depth = self.fitting_function(new_depth_range, *params)
        #             # vis_list.append(vis_depth)
        #             # vis_range.append(new_depth_range)
                    
        return depth_peak_illum_idx
    
    # fitting function
    def fitting_function(self, x, a, b, c):
        """
            non-linear fitting function
        """
        return a*(x**b) + c
    
    # bring depth values
    def get_depth(self, position):
        """
            bring depth values (mm) for each spectralon position
        """
        depth_dir = os.path.join(self.to_depth_dir%(self.date, position), "2023%s_spectralon_%s.npy"%(self.date, position))
        depth = np.load(depth_dir)[:,:,2].reshape(self.cam_H * self.cam_W) # only get z(=depth) value

        return depth
    
    # get depth dependent peak illumination indices
    def get_depth_peak_illum(self):
        """
            get depth dependent peak illumination index of zero orders and first orders
            
            shape :
            depth(600mm-900mm at 1mm interval), 2(m=-1 or 1 and 0), wvl(430nm, 600nm - 660nm), sample pts
        """
        depth_peak_illum_idx = self.depth_interpolation() # 301, 3, wvls, sample_pts
        np.save(os.path.join(self.npy_dir,'./depth_peak_illum_idx.npy'), depth_peak_illum_idx)
        depth_peak_illum_idx = np.load(os.path.join(self.npy_dir,'./depth_peak_illum_idx.npy'))
        
                                                    # depth, m order (-1 or 1 and zero), wvl(430, 660), pts
        depth_peak_illum_idx_final = np.zeros(shape=(len(self.depth_arange), 2, len(self.wvl_list), self.sample_pts.shape[0])) 
        
        # masking out to get +1 or -1
        # difference between 430nm, 660nm for +1 and -1
        mfirst_diff = abs(depth_peak_illum_idx[:,0,0] - depth_peak_illum_idx[:,0,-1])
        pfirst_diff = abs(depth_peak_illum_idx[:,2,0] - depth_peak_illum_idx[:,2,-1])

        pfirst_diff = pfirst_diff[:,np.newaxis,:]
        mfirst_diff = mfirst_diff[:,np.newaxis,:]

        # Further broadcast the masks to the shape of max_data
        pfirst_diff = np.repeat(pfirst_diff, 7, axis=1)
        mfirst_diff = np.repeat(mfirst_diff, 7, axis=1)

        # if difference of 430nm and 660nm is larger, take that first order
        mask_pfirst = mfirst_diff <= pfirst_diff        
        depth_peak_illum_idx_final[:,1][mask_pfirst] = depth_peak_illum_idx[:,2][mask_pfirst]
        depth_peak_illum_idx_final[:,1][mask_pfirst] = depth_peak_illum_idx[:,2][mask_pfirst]

        mask_mfirst = mfirst_diff > pfirst_diff
        depth_peak_illum_idx_final[:,1][mask_mfirst] = depth_peak_illum_idx[:,0][mask_mfirst]
        depth_peak_illum_idx_final[:,1][mask_mfirst] = depth_peak_illum_idx[:,0][mask_mfirst]
        # input zero order
        depth_peak_illum_idx_final[:,0] = depth_peak_illum_idx[:,1].mean(axis = 1)[:,np.newaxis,:]
        
        return depth_peak_illum_idx_final # depth, m order (-1 or 1 and zero), wvl(430, 600 - 660), pts
        
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    date = arg.calibrated_date
    
    # front_peak_illum_idx = DataProcess(arg, date, "front").get_first_idx()
    # mid_peak_illum_idx = DataProcess(arg, date, "mid").get_first_idx()
    # mid2_peak_illum_idx = DataProcess(arg, date, "mid2").get_first_idx()
    # mid3_peak_illum_idx = DataProcess(arg, date, "mid3").get_first_idx()
    # back_peak_illum_idx = DataProcess(arg, date, "back").get_first_idx()
    
    front_peak_illum_idx = np.load("./dataset/image_formation/2023%s/npy_data/peak_illum_idx_front.npy"%date)
    mid_peak_illum_idx = np.load("./dataset/image_formation/2023%s/npy_data/peak_illum_idx_mid.npy"%date)
    mid2_peak_illum_idx = np.load("./dataset/image_formation/2023%s/npy_data/peak_illum_idx_mid2.npy"%date)
    mid3_peak_illum_idx = np.load("./dataset/image_formation/2023%s/npy_data/peak_illum_idx_mid3.npy"%date)
    back_peak_illum_idx = np.load("./dataset/image_formation/2023%s/npy_data/peak_illum_idx_back.npy"%date)
    
    DepthInterpolation(arg, date, front_peak_illum_idx, mid_peak_illum_idx, mid2_peak_illum_idx, mid3_peak_illum_idx, back_peak_illum_idx).get_depth_peak_illum()