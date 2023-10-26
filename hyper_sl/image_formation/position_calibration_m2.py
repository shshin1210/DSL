import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
from hyper_sl.utils.ArgParser import Argument
import numpy as np
import matplotlib.pyplot as plt
from data_process import DataProcess
from depth_interpolation_m2 import DepthInterpolation
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

    
    def interpolate_image(self, depth_peak_illum_idx):
        """
            make a HxW image with 20 sample points using interpolation method
        """

        # depth, m, H, W, wvl
        # m = +1 xy interpolation
        depth_peak_illum_idx_image = np.zeros(shape=(len(self.depth_arange), len(self.m_list), len(self.wvl_list), self.cam_H, self.cam_W))
        
        for d in range(len(self.depth_arange)):
            for m in range(len(self.m_list)):
                for w in range(len(self.wvl_list)):
                    depth_peak_illum_idx_intp = self.interpolation(self.sample_pts, depth_peak_illum_idx[d,m,w])
                    depth_peak_illum_idx_image[d, m, w] = depth_peak_illum_idx_intp
                    
        return depth_peak_illum_idx_image

        
    def full_wavelength_interpolation(self, depth_peak_illum_idx):
        """
            interpolate for 430nm ~ 660nm 5 nm interval
        """
        # depth_peak_illum_idx_image = self.interpolate_image(depth_peak_illum_idx)
        # np.save(os.path.join(self.npy_dir, 'depth_peak_illum_idx_image.npy'), depth_peak_illum_idx_image)
        depth_peak_illum_idx_image = np.load(os.path.join(self.npy_dir, 'depth_peak_illum_idx_image.npy'))
        
        print('debug')
        depth_peak_illum_idx_image_final = np.zeros(shape=(len(self.depth_arange), 2, len(self.wvl_list), self.cam_H, self.cam_W))
        
        mfirst_diff = abs(depth_peak_illum_idx_image[:,0,0] - depth_peak_illum_idx_image[:,0,-1])        
        pfirst_diff = abs(depth_peak_illum_idx_image[:,2,0] - depth_peak_illum_idx_image[:,2,-1])        
        
        mfirst_mask = mfirst_diff > pfirst_diff # take m = -1
        pfirst_mask = mfirst_diff <= pfirst_diff # take m = +1
        
        # make 7 wvl
        mfirst_mask = mfirst_mask[:,np.newaxis,:]
        pfirst_mask = pfirst_mask[:,np.newaxis,:]
        
        mfirst_mask = np.repeat(mfirst_mask, 7, axis=1)
        pfirst_mask = np.repeat(pfirst_mask, 7, axis=1)
        
        depth_peak_illum_idx_image_final[:,1][mfirst_mask] = depth_peak_illum_idx_image[:,0][mfirst_mask]
        depth_peak_illum_idx_image_final[:,1][pfirst_mask] = depth_peak_illum_idx_image[:,2][pfirst_mask]

        depth_peak_illum_idx_image_final[:,0] = depth_peak_illum_idx_image[:,1]
        
        first_illum_idx = depth_peak_illum_idx_image_final[:,1].reshape(len(self.depth_arange), len(self.wvl_list), self.cam_H * self.cam_W) # 301, 7, H, W
        first_illum_idx_final = np.zeros(shape=(len(self.depth_arange), len(self.new_wvls), self.cam_H*self.cam_W), dtype = np.int16)
        
        for i in range(self.cam_H * self.cam_W):
            if (0 in first_illum_idx[:,:,i]) or (317 in first_illum_idx[:,:,i]):                
                # 317이 있는 depth value와 wavelength 의 위치
                depth_idx_0 = np.where(first_illum_idx[:,:,i] == 0.)[0] # depths
                depth_idx_317 = np.where(first_illum_idx[:,:,i] == 317.)[0] # depths

                if len(depth_idx_0) * len(depth_idx_317) == 0:
                    if len(depth_idx_0) > 0:
                        depth_unique_val, depth_first_idx = np.unique(depth_idx_0, return_index=True)
                        depth_wvl_idx = np.where(first_illum_idx[:,:,i] == 0.)[1][depth_first_idx]
                        valid_idx = depth_wvl_idx -1
                        wvl_idx = np.searchsorted(self.new_wvls*1e9 , self.wvl_list[valid_idx])
                    
                    else:
                        depth_unique_val, depth_first_idx = np.unique(depth_idx_317, return_index=True)
                        depth_wvl_idx = np.where(first_illum_idx[:,:,i] == 317.)[1][depth_first_idx]
                        valid_idx = depth_wvl_idx -1
                        wvl_idx = np.searchsorted(self.new_wvls*1e9 , self.wvl_list[valid_idx])
                        
                     # for depths with 0 or 317
                    linspace_arrays = np.array([np.linspace(start, stop, num) for start, stop, num in zip(first_illum_idx[depth_unique_val,0,i], first_illum_idx[depth_unique_val,valid_idx,i], wvl_idx +1)])
                    interval_array_tmp = np.zeros(shape=(len(self.depth_arange),len(self.new_wvls)))               

                    for k in range(len(wvl_idx)):
                        interval_array_tmp[depth_unique_val[k],:wvl_idx[k]+1] = linspace_arrays[k]
                        interval_array_tmp[depth_unique_val[k],wvl_idx[k]+1:] = linspace_arrays[k][-1]
                    
                    interval_array_tmp = np.round(interval_array_tmp).astype(np.int16)
                    first_illum_idx_final[depth_unique_val,:,i] = interval_array_tmp[depth_unique_val]
                    
                    # for depths with none of 0 or 317
                    interval_array_none = np.linspace(first_illum_idx[:depth_unique_val[0],0,i], first_illum_idx[:depth_unique_val[0],-1,i], 47)
                    interval_array_none = np.round(interval_array_none).astype(np.int16)
                    first_illum_idx_final[:depth_unique_val[0],:,i] = interval_array_none.T

                else:
                    # for 0
                    depth_unique_val, depth_first_idx = np.unique(depth_idx_0, return_index=True)
                    depth_wvl_idx = np.where(first_illum_idx[:,:,i] == 0.)[1][depth_first_idx]
                    valid_idx = depth_wvl_idx -1
                    wvl_idx = np.searchsorted(self.new_wvls*1e9 , self.wvl_list[valid_idx])
                    
                    # for depths with 0
                    linspace_arrays = np.array([np.linspace(start, stop, num) for start, stop, num in zip(first_illum_idx[depth_unique_val,0,i], first_illum_idx[depth_unique_val,valid_idx,i], wvl_idx +1)])
                    interval_array_tmp = np.zeros(shape=(len(self.depth_arange),len(self.new_wvls)))               

                    for k in range(len(wvl_idx)):
                        interval_array_tmp[depth_unique_val[k],:wvl_idx[k]+1] = linspace_arrays[k]
                        interval_array_tmp[depth_unique_val[k],wvl_idx[k]+1:] = linspace_arrays[k][-1]
                    
                    interval_array_tmp = np.round(interval_array_tmp).astype(np.int16)
                    first_illum_idx_final[depth_unique_val,:,i] = interval_array_tmp[depth_unique_val]
                    
                    # for 317
                    depth_unique_val, depth_first_idx = np.unique(depth_idx_317, return_index=True)
                    depth_wvl_idx = np.where(first_illum_idx[:,:,i] == 317.)[1][depth_first_idx]
                    valid_idx = depth_wvl_idx -1
                    wvl_idx = np.searchsorted(self.new_wvls*1e9 , self.wvl_list[valid_idx])

                    # for depths with 317
                    linspace_arrays = np.array([np.linspace(start, stop, num) for start, stop, num in zip(first_illum_idx[depth_unique_val,0,i], first_illum_idx[depth_unique_val,valid_idx,i], wvl_idx +1)])
                    interval_array_tmp = np.zeros(shape=(len(self.depth_arange),len(self.new_wvls)))               

                    for k in range(len(wvl_idx)):
                        interval_array_tmp[depth_unique_val[k],:wvl_idx[k]+1] = linspace_arrays[k]
                        interval_array_tmp[depth_unique_val[k],wvl_idx[k]+1:] = linspace_arrays[k][-1]
                    
                    interval_array_tmp = np.round(interval_array_tmp).astype(np.int16)
                    first_illum_idx_final[depth_unique_val,:,i] = interval_array_tmp[depth_unique_val]
                
            # No depths with 0 or 317
            else:
                interval_array = np.linspace(first_illum_idx[:,0,i], first_illum_idx[:,-1,i], 47)
                interval_array = np.round(interval_array).astype(np.int16)
                first_illum_idx_final[:,:,i] = interval_array.T

        first_illum_idx_final = np.array(first_illum_idx_final)
        np.save(os.path.join(self.npy_dir, 'first_illum_idx_final_transp_%s.npy'%"test_2"), first_illum_idx_final)


        # # no vectorization
        # for d in range(len(self.depth_arange)):
        #     for i in range(self.cam_H * self.cam_W):
        #         if (0 in first_illum_idx[d,:,i]) or (317 in first_illum_idx[d,:,i]):
        #             idx_0 = np.where(first_illum_idx[d,:,i] == 0.)[0]
        #             idx_317 = np.where(first_illum_idx[d,:,i] == 317.)[0]
                    
        #             if len(idx_0) > 0:
        #                 valid_idx = idx_0[0] -1
        #                 wvl_idx = np.where(self.new_wvls*1e9 == self.wvl_list[valid_idx])[0][0]
        #             else:
        #                 valid_idx = idx_317[0] -1
        #                 wvl_idx = np.where(self.new_wvls*1e9 == self.wvl_list[valid_idx])[0][0]

        #             interval_array = np.linspace(first_illum_idx[d,0,i], first_illum_idx[d,valid_idx,i], wvl_idx +1) # len(wvl_idx + 1)
        #             interval_array_tmp = np.zeros(len(self.new_wvls))
        #             interval_array_tmp[:wvl_idx+1] = interval_array
        #             interval_array_tmp[wvl_idx+1:] = interval_array[wvl_idx]
                    
        #             interval_array_tmp = np.round(interval_array_tmp).astype(np.int16)
        #             first_illum_idx_final[d,:,i] = interval_array_tmp

        #         else:
        #             interval_array = np.linspace(first_illum_idx[d,0,i], first_illum_idx[d,-1,i], 47)
        #             interval_array = np.round(interval_array).astype(np.int16)
        #             first_illum_idx_final[d,:,i] = interval_array

        # first_illum_idx_final = np.array(first_illum_idx_final)

        # first_illum_idx_final_reshape = first_illum_idx_final.reshape(self.cam_H, self.cam_W, len(self.depth_arange), len(self.new_wvls))

        # # to make 47, cam_H, cam_W, 3 array
        # first_illum_idx_final_transp = first_illum_idx_final_reshape.transpose(3,2,0,1)
        
        # np.save(os.path.join(self.npy_dir, 'first_illum_idx_final_transp_%s.npy'%"test_2"), first_illum_idx_final_transp)
        # return first_illum_idx_final_transp
    
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
    
    depth_peak_illum_idx = DepthInterpolation(arg, date, front_peak_illum_idx, mid_peak_illum_idx, mid2_peak_illum_idx, mid3_peak_illum_idx, back_peak_illum_idx).get_depth_peak_illum()
    
    PositionCalibration().full_wavelength_interpolation(depth_peak_illum_idx)