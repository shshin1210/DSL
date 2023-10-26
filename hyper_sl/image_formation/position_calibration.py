import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
from hyper_sl.utils.ArgParser import Argument
import numpy as np
import matplotlib.pyplot as plt
from data_process import DataProcess
from depth_interpolation import DepthInterpolation
from scipy import interpolate
from scipy.interpolate import griddata, NearestNDInterpolator


class PositionCalibration():
    def __init__(self):
        # arguments
        self.date = date
        self.depth = arg.depth_list
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        
        self.m_list = arg.m_list
        self.new_wvls = np.linspace(430*1e-9, 660*1e-9, 47)
        self.wvl_list = np.array([430, 600, 610, 620, 640, 650, 660])
        self.depth_start = 600
        self.depth_end = 900     
        self.depth_arange = np.arange(self.depth_start, self.depth_end + 1, 1)
        
        # self.sample_pts = np.array([[10 + i*120, 50 + j*51] for j in range(10) for i in range(8)])
        self.sample_pts = np.array([[10 + i*60, 50 + j*51] for j in range(10) for i in range(15)])
        # self.sample_pts = np.array([[10 + i, 50 + j] for j in range(461) for i in range(841)])
        self.sample_pts_flatt = np.array([[self.sample_pts[i,0]+self.sample_pts[i,1]*self.cam_W] for i in range(self.sample_pts.shape[0])]).squeeze()
        
        # dir
        self.npy_dir = "./dataset/image_formation/2023%s/npy_data"%date
    
    
    def interpolation(self, sample_pts, data):
        """
            interpolate 430nm, 660nm illum index values for each depth
        
        """
        # Create a mesh grid for the image dimensions
        grid_y, grid_x = np.mgrid[0:self.cam_H, 0:self.cam_W]

        # Use griddata to interpolate
        interp_values = griddata(sample_pts, data, (grid_x, grid_y), method='cubic')

        # Step 2: Enhance the set of points and values using interpolated results
        y_coords, x_coords = np.where(~np.isnan(interp_values))

        enhanced_points = np.column_stack((y_coords, x_coords))
        enhanced_values = interp_values[~np.isnan(interp_values)]

        # Step 3: Extrapolate using the enhanced set of points and values
        interpolator = NearestNDInterpolator(enhanced_points, enhanced_values)

        extrapolated_values = interpolator(grid_y, grid_x)

        return extrapolated_values
    
    # make -1 orders to be +1 order illumination index for smooth interpolation
    def make_new_peak_illum_idx(self, depth_peak_illum_idx_final, diffs):
        """
            make all -1 & 1 orders to +1 order (preprocessing before interpolation)
            depth_peak_illum_idx_final : depth, 2(0, first), wvl(430, 660), # px
        """

        diffs = diffs.reshape(-1, self.cam_H * self.cam_W)[:,self.sample_pts_flatt]
        new_peak_illum_idx = np.zeros(shape=(len(self.depth_arange), 2, 2, self.sample_pts.shape[0])) # depth arange, m(0 or first), wvl(430, 660), # pts

        # pick +1 order or -1 (if -1 order, make its illum idx as +1 order)
        for d in range(len(self.depth_arange)):
            for i in range(len(self.sample_pts_flatt)):
                difference_zero_430nm = abs(depth_peak_illum_idx_final[d,0,0,i] - depth_peak_illum_idx_final[d,1,0,i])
                
                if depth_peak_illum_idx_final[d,0,0,i] > depth_peak_illum_idx_final[d,1,0,i]: # mfirst
                    new_peak_illum_idx[d,1,0,i] = depth_peak_illum_idx_final[d,0,0,i] + (difference_zero_430nm)
                    new_peak_illum_idx[d,1,1,i] = depth_peak_illum_idx_final[d,0,0,i] + (difference_zero_430nm) + diffs[d,i]
                else: # pfirst
                    new_peak_illum_idx[d,1,0,i] = depth_peak_illum_idx_final[d,1,0,i] + 1 # 430nm
                    new_peak_illum_idx[d,1,1,i] = depth_peak_illum_idx_final[d,1,0,i] + 2 + diffs[d,i] 
                                            #   depth_peak_illum_idx_final[d,1,-1,i] # 660nm
                    
                new_peak_illum_idx[d,0,0,i] = depth_peak_illum_idx_final[d,0,0,i]
                new_peak_illum_idx[d,0,1,i] = depth_peak_illum_idx_final[d,0,-1,i]

        return new_peak_illum_idx
    
    def original_illum_idx(self, depth_new_peak_image_illum_idx):
        """
            divide it to valid order / illum index
            new : new_peak_image_illum_idx_re
            orig : orig_peak_image_illum_idx

            if new index 430nm and 660nm difference is larger pick new index
            if not pick original index 
        """
        orig_peak_image_illum_idx = np.zeros_like(depth_new_peak_image_illum_idx).reshape(len(self.depth_arange), 2, self.cam_H*self.cam_W, 2)

        for i in range(self.cam_H * self.cam_W):
            orig_peak_image_illum_idx[:,1,i,0] = depth_new_peak_image_illum_idx[:,0,i,0] - abs(depth_new_peak_image_illum_idx[:,0,i,0] - (depth_new_peak_image_illum_idx[:,1,i,0]-1))
            orig_peak_image_illum_idx[:,1,i,1] = depth_new_peak_image_illum_idx[:,0,i,0] - abs(depth_new_peak_image_illum_idx[:,0,i,0] - (depth_new_peak_image_illum_idx[:,1,i,1]-1))
        
        orig_peak_image_illum_idx[:,0] = depth_new_peak_image_illum_idx[:,0]
        
        return orig_peak_image_illum_idx
    
    # difference between 430nm and 660nm
    def interpolate_diffs(self, depth_peak_illum_idx_final):
        """
            interpolate the difference between 430nm and 630nm
        
        """
        diffs = np.zeros(shape=(len(self.depth_arange), self.cam_H, self.cam_W))
        
        for d in range(len(self.depth_arange)):
            
            # 크기 : sample points / depth, m, wvl, HxW
            difference_430nm_660nm = abs(depth_peak_illum_idx_final[d,1,0] - depth_peak_illum_idx_final[d,1,-1]) + 1

            # depth, m, wvl, sample pts
            # 만약에 반복되는 숫자가 있다면? cnt 0 and 318
            for i in range(len(self.sample_pts_flatt)):
                cnt_0 = np.count_nonzero(np.round(depth_peak_illum_idx_final[d,1,:,i]).astype(np.int16) == 0)
                cnt_317 = np.count_nonzero(np.round(depth_peak_illum_idx_final[d,1,:,i]).astype(np.int16) == 317)
                
                if (cnt_0 > 0) or (cnt_317 > 0):
                    cnt_0 = (cnt_0 - 1)*2
                    cnt_317 *= (cnt_317 - 1)*2
                    
                difference_430nm_660nm[i] += (cnt_0 + cnt_317)
                
            intp_diff = self.interpolation(self.sample_pts, difference_430nm_660nm)
            diffs[d] = intp_diff
            
        return diffs
    
    def interpolate_image(self, depth_peak_illum_idx_final):
        """
            make a HxW image with 20 sample points using interpolation method
        """
        # difference between 430nm and 660nm
        diffs = self.interpolate_diffs(depth_peak_illum_idx_final)
        np.save(os.path.join(self.npy_dir,'diffs.npy'), diffs)
        # diffs = np.load(os.path.join(self.npy_dir,'diffs.npy'))
        
        # All m = +1 illum index
        new_peak_illum_idx = self.make_new_peak_illum_idx(depth_peak_illum_idx_final, diffs)

        # depth, m, H, W, wvl
        # m = +1 xy interpolation
        depth_new_peak_image_illum_idx = np.zeros(shape=(len(self.depth_arange), 2, self.cam_H, self.cam_W, 2))
        
        for d in range(len(self.depth_arange)):
            new_peak_image_illum_idx_first_430nm = self.interpolation(self.sample_pts, new_peak_illum_idx[d,1,0])
            new_peak_image_illum_idx_first_660nm = self.interpolation(self.sample_pts, new_peak_illum_idx[d,1,1])
            new_peak_image_illum_idx_first = np.stack((new_peak_image_illum_idx_first_430nm, new_peak_image_illum_idx_first_660nm), axis = 2)
            
            new_peak_image_illum_idx_zero_430nm = self.interpolation(self.sample_pts, new_peak_illum_idx[d,0,0])
            new_peak_image_illum_idx_zero = np.stack((new_peak_image_illum_idx_zero_430nm, new_peak_image_illum_idx_zero_430nm), axis = 2)
            
            depth_new_peak_image_illum_idx[d,1] = new_peak_image_illum_idx_first
            depth_new_peak_image_illum_idx[d,0] = new_peak_image_illum_idx_zero

        return depth_new_peak_image_illum_idx

    def pick_valid(self, depth_peak_illum_idx_final):
        """
            pick the valid order -1(original) or +1(new)
        """
        depth_new_peak_image_illum_idx = self.interpolate_image(depth_peak_illum_idx_final).reshape(len(self.depth_arange), 2, self.cam_H * self.cam_W, 2)
        np.save(os.path.join(self.npy_dir,'depth_new_peak_image_illum_idx.npy'), depth_new_peak_image_illum_idx)
        # depth_new_peak_image_illum_idx = np.load(os.path.join(self.npy_dir,'./depth_new_peak_image_illum_idx.npy'))
        
        # All m = -1 illum index
        orig_peak_image_illum_idx = self.original_illum_idx(depth_new_peak_image_illum_idx)
                
        peak_image_illum_idx = np.zeros_like(depth_new_peak_image_illum_idx)

        depth_new_peak_image_illum_idx_cp = depth_new_peak_image_illum_idx.copy()
        orig_peak_image_illum_idx_cp = orig_peak_image_illum_idx.copy()
        
        # vectorization                  
        depth_new_peak_image_illum_idx_cp[:,1,:,0][depth_new_peak_image_illum_idx[:,1,:,0] > 317] = 317
        depth_new_peak_image_illum_idx_cp[:,1,:,1][depth_new_peak_image_illum_idx[:,1,:,0] > 317] = 317
        depth_new_peak_image_illum_idx_cp[:,1,:,1][depth_new_peak_image_illum_idx[:,1,:,1] > 317] = 317

        orig_peak_image_illum_idx_cp[:,1,:,0][orig_peak_image_illum_idx[:,1,:,0] < 0] = 0
        orig_peak_image_illum_idx_cp[:,1,:,1][orig_peak_image_illum_idx[:,1,:,0] < 0] = 0
        orig_peak_image_illum_idx_cp[:,1,:,1][orig_peak_image_illum_idx[:,1,:,1] < 0] = 0

        # # depth, m, HxW, wvl
        new_diff = abs(depth_new_peak_image_illum_idx_cp[:,1,:,0] - depth_new_peak_image_illum_idx_cp[:,1,:,1])
        orig_diff = abs(orig_peak_image_illum_idx_cp[:,1,:,0] - orig_peak_image_illum_idx_cp[:,1,:,1])
        
        mask_1 = new_diff > orig_diff
        peak_image_illum_idx[:,1,:,0][mask_1] = depth_new_peak_image_illum_idx[:,1,:,0][mask_1]
        peak_image_illum_idx[:,1,:,1][mask_1] = depth_new_peak_image_illum_idx[:,1,:,1][mask_1]

        mask_2 = new_diff <= orig_diff
        peak_image_illum_idx[:,1,:,0][mask_2] = orig_peak_image_illum_idx[:,1,:,0][mask_2]
        peak_image_illum_idx[:,1,:,1][mask_2] = orig_peak_image_illum_idx[:,1,:,1][mask_2]

        peak_image_illum_idx[:,0] = depth_new_peak_image_illum_idx[:,0]
        
        np.save(os.path.join(self.npy_dir,'./peak_image_illum_idx.npy'), peak_image_illum_idx)
        
        return peak_image_illum_idx
        
    def full_wavelength_interpolation(self, depth_peak_illum_idx_final):
        """
            interpolate for 430nm ~ 660nm 5 nm interval
        """
        peak_image_illum_idx = self.pick_valid(depth_peak_illum_idx_final)
        np.save(os.path.join(self.npy_dir, 'peak_image_illum_idx.npy'), peak_image_illum_idx)
        # peak_image_illum_idx = np.load('./peak_image_illum_idx_test.npy')
        
        position_430nm = peak_image_illum_idx[:,1,:,0]
        position_660nm = peak_image_illum_idx[:,1,:,1]

        start_idx = position_430nm
        end_idx = position_660nm

        first_illum_idx_final = np.zeros(shape=(self.cam_H*self.cam_W, len(self.depth_arange), len(self.new_wvls)), dtype = np.int16)
        for i in range(self.cam_H* self.cam_W):
            interval_array = np.linspace(start_idx[:,i], end_idx[:,i], 47).T
            interval_array = np.round(interval_array).astype(np.int16)
            first_illum_idx_final[i] = interval_array
        first_illum_idx_final = np.array(first_illum_idx_final)

        first_illum_idx_final_reshape = first_illum_idx_final.reshape(self.cam_H, self.cam_W, len(self.depth_arange), len(self.new_wvls))

        # to make 47, cam_H, cam_W, 3 array
        first_illum_idx_final_transp = first_illum_idx_final_reshape.transpose(3,2,0,1)
        
        np.save(os.path.join(self.npy_dir, 'first_illum_idx_final_transp_%s.npy'%"test_2"), first_illum_idx_final_transp)
        return first_illum_idx_final_transp
    
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
    
    depth_peak_illum_idx_final = DepthInterpolation(arg, date, front_peak_illum_idx, mid_peak_illum_idx, mid2_peak_illum_idx, mid3_peak_illum_idx, back_peak_illum_idx).get_depth_peak_illum()
    
    PositionCalibration().full_wavelength_interpolation(depth_peak_illum_idx_final)