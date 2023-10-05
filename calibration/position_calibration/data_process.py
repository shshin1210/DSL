import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
import matplotlib.pyplot as plt


class DataProcess():
    def __init__(self, arg, date, position):
        
        # arguments
        self.date = date
        self.position = position
        self.peak_wvl = np.array([430, 660])
        self.n_illum = 318 # arg.illum_num 나중에 argparser에서 illum_dir 바꾸기!!!!!!!!!!!!!!!!!!
        self.cam_H = arg.cam_H
        self.cam_W = arg.cam_W
        
        # dir
        self.data_dir = "./calibration/position_calibration/2023%s/%s_depth/"%(date, position)
        self.depth_dir = "./calibration/position_calibration/2023%s/%s_depth/spectralon/2023%s_spectralon_%s"%(date, position,date, position)
        self.npy_dir = "./calibration/position_calibration/2023%s/npy_data"%date
        
    def get_data(self):
        wvl_imgs = []
        for i in range(self.n_illum):
            wvl_dir = os.path.join(self.data_dir, "%dnm/calibration00/capture_%04d.png")           
            wvl_img = np.array([cv2.imread(wvl_dir%(w, i), -1)[:,:,::-1] for w in self.peak_wvl]) / 65535.
            
            # if self.position == "back": # back position needs black image to be subtracted
            #     black_dir = os.path.join(self.data_dir, "black/calibration00/black_%dnm.png")
            #     black_imgs = np.array([cv2.imread(black_dir%w, -1)[:,:,::-1] for w in self.peak_wvl]) / 65535.
            #     wvl_img = abs(wvl_img - black_imgs)

            wvl_imgs.append(wvl_img)
        
        wvl_imgs = np.array(wvl_imgs)
        wvl_imgs = wvl_imgs.transpose(1,0,2,3,4) # 2, 318, 580, 890, 3
        
        return wvl_imgs
    
    def get_max_data(self):
        """
            get maximum rgb value for each value
            
            returns : gray value image shaped wvl(430m, 660nm), illum index, H, W
        
        """
        wvl_imgs = self.get_data() # 2, 318, 580, 890, 3 (wvl, illum, H, W, 3)
        
        # put intensity data
        max_data = np.zeros(shape=(len(self.peak_wvl), self.n_illum, self.cam_H * self.cam_W))
        
        # save data for wvls
        # max value defined for each illumination pattern 
        for w in range(len(self.peak_wvl)):
            for l in range(self.n_illum):
                wvl_imgs_reshaped = wvl_imgs[w, l].reshape(-1, 3)
                max_idx = np.argmax(wvl_imgs_reshaped, axis = 1) # rgb 채널 중 max value의 channel 찾기
                max_data[w, l] = np.array([wvl_imgs_reshaped[i, max_idx[i]] for i in range(self.cam_H * self.cam_W)]) # 2, illum index, H, W

        return max_data
    
    
    def get_zero_idx(self, max_data):
        """
            get zero illumination index for each pixels
        
        """        
        zero_illum_idx = np.zeros(shape=(len(self.peak_wvl), self.cam_H * self.cam_W))
        
        for w in range(len(self.peak_wvl)):
            for i in range(self.cam_H*self.cam_W):
                max_idx = np.argmax(max_data[w,:,i])
                zero_illum_idx[w, i] = max_idx
        
        zero_illum_idx_final = np.round(zero_illum_idx.mean(axis = 0))

        return zero_illum_idx_final
    
    def get_first_idx(self):
        """
            get first order index (only one (+1 or -1))
            
        """
        # args
        patch_size = 30
        
        max_data = self.get_max_data()
        np.save(os.path.join(self.npy_dir, 'max_data_%s.npy'%self.position), max_data)
        max_data = np.load(os.path.join(self.npy_dir, 'max_data_%s.npy'%self.position))
        
        peak_illum_idx = np.zeros(shape=(3, len(self.peak_wvl), self.cam_H * self.cam_W))
        peak_illum_idx[1] = self.get_zero_idx(max_data)[np.newaxis,:]
                
        for w in range(len(self.peak_wvl)):
            for i in range(self.cam_H*self.cam_W):
                # 2, illum index, H, W
                idx_mfirst = (peak_illum_idx[1,0,i] - patch_size).astype(np.int32)
                idx_pfirst = (peak_illum_idx[1,0,i] + patch_size).astype(np.int32)
                
                if peak_illum_idx[1,0,i] - patch_size <= 0:
                    idx_mfirst = 1
                elif peak_illum_idx[1,0,i] + patch_size >= 317:
                    idx_pfirst = 316
                
                # zero order 제외 max값을 가지는 illum idx 넣기
                max_idx_mfirst = np.argmax(max_data[w, :idx_mfirst, i])
                max_idx_pfirst = np.argmax(max_data[w, idx_pfirst:, i]) + idx_pfirst

                # 여기서 intensity 비교
                if max_data[w, max_idx_mfirst, i] < 0.06:
                    max_idx_mfirst = 0
                if max_data[w, max_idx_pfirst, i] < 0.06:
                    max_idx_pfirst = 318
                    
                peak_illum_idx[0,w,i] = max_idx_mfirst
                peak_illum_idx[2,w,i] = max_idx_pfirst
        
        # save peak illumination index data
        np.save(os.path.join(self.npy_dir, 'peak_illum_idx_%s'%self.position), peak_illum_idx)
        
        return peak_illum_idx
        
                
if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
    date = "0922"
    position = "back"
    
    peak_illum_idx = DataProcess(arg, date, position).get_first_idx()