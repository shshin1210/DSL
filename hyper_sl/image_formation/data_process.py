import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np

"""
Data processing

Process single depth spectralon data captured with 430nm, 600nm - 660nm band pass filter

DataProcess 
    input : date (date of the calibration datas), position (depth position of spectralon)
    output : zero, first order illumination index / shape : 3(m), 2(wvls), 580*890
    
def get_data : bring depth / wvls data image

def get_max_data : get the max intensity between RGB channel (ex. 430nm will take blue channel)

def get_zero_idx : get the zero order illumination index by argmax

def get_first_idx : get -1 order illumination index and +1 order illumination index

"""

class DataProcess():
    def __init__(self, arg, date, position):
        
        # arguments
        self.date = date
        self.position = position
        # self.wvl_list = np.array([430, 660])
        self.wvl_list = np.array([430, 600, 610, 620, 640, 650, 660])
        self.n_illum = arg.illum_num
        self.cam_H = arg.cam_H
        self.cam_W = arg.cam_W
        
        # dir
        self.data_dir = "./dataset/image_formation/2023%s/%s_depth"%(self.date, self.position)
        self.npy_dir = "./dataset/image_formation/2023%s/npy_data"%date
        
    def get_data(self):
        """
            bring depth / wvls data image
            
            output : 2(wvls), 318, 580, 890, 3(rgb)
            
            wvls dimention 2 -> 6? (430, 600, 610, 620, 640, 650, 660)
            
        """
        wvl_imgs = []
        for i in range(self.n_illum):
            wvl_dir = os.path.join(self.data_dir, "%dnm/calibration00/capture_%04d.png")           
            wvl_img = np.array([cv2.imread(wvl_dir%(w, i), -1)[:,:,::-1] for w in self.wvl_list]) / 65535.
            wvl_imgs.append(wvl_img)
        
        wvl_imgs = np.array(wvl_imgs)
        wvl_imgs = wvl_imgs.transpose(1,0,2,3,4) # wvls, 318, 580, 890, rgb
        
        return wvl_imgs
    
    def get_max_data(self):
        """
            get maximum rgb value for each value
            
            returns : gray value image shape : 2 wvl(430m, 660nm), 318 illum index, H, W
        
        """
        wvl_imgs = self.get_data() # wvl, 318, 580, 890, 3 (wvl, illum, H, W, 3)
        
        # put intensity data
        max_data = np.zeros(shape=(len(self.wvl_list), self.n_illum, self.cam_H * self.cam_W))
        
        # save data for wvls
        # max value defined for each illumination pattern 
        for w in range(len(self.wvl_list)):
            for l in range(self.n_illum):
                wvl_imgs_reshaped = wvl_imgs[w, l].reshape(-1, 3)
                max_idx = np.argmax(wvl_imgs_reshaped, axis = 1) # rgb 채널 중 max value의 channel 찾기
                max_data[w, l] = np.array([wvl_imgs_reshaped[i, max_idx[i]] for i in range(self.cam_H * self.cam_W)]) # 2, illum index, H, W

        return max_data
    
    
    def get_zero_idx(self, max_data):
        """
            get zero illumination index for each pixels
        
        """        
        zero_illum_idx = np.zeros(shape=(len(self.wvl_list), self.cam_H * self.cam_W))
        
        for w in range(len(self.wvl_list)):
            for i in range(self.cam_H*self.cam_W):
                max_idx = np.argmax(max_data[w,:,i])
                zero_illum_idx[w, i] = max_idx
        
        zero_illum_idx_final = np.round(zero_illum_idx.mean(axis = 0))

        return zero_illum_idx_final
    
    def get_first_idx(self):
        """
            get -1 order illumination index and +1 order illumination index
            
            output : 3(m), 2(wvl), 580*890
        """
        # args
        patch_size = 30
        
        max_data = self.get_max_data()
        
        # make all intensity in a valid range
        for w_idx in range(len(self.wvl_list)):
            for idx, i in enumerate(self.sample_pts_flatt):
                if (np.median(max_data[w_idx,:,i]) > 0.07) and (np.median(max_data[w_idx,:,i]) < 0.09):
                    max_data[w_idx,:,i] = max_data[w_idx,:,i] - 0.02
                if (np.median(max_data[w_idx,:,i]) >= 0.09):
                    max_data[w_idx,:,i] = max_data[w_idx,:,i] - 0.03
                if(max_data[w_idx,:,i].min() <= 0.06):
                    max_data[w_idx,:,i] = max_data[w_idx,:,i] + 0.008
                
        np.save(os.path.join(self.npy_dir, 'max_data_%s.npy'%self.position), max_data)
        # max_data = np.load(os.path.join(self.npy_dir, 'max_data_%s.npy'%self.position))
        
        peak_illum_idx = np.zeros(shape=(3, len(self.wvl_list), self.cam_H * self.cam_W))
        peak_illum_idx[1] = self.get_zero_idx(max_data)[np.newaxis,:] # put zero illumination index
                
        for w in range(len(self.wvl_list)):
            for i in range(self.cam_H*self.cam_W):
                # 2, illum index, H, W
                idx_mfirst = (peak_illum_idx[1,0,i] - patch_size).astype(np.int32)
                idx_pfirst = (peak_illum_idx[1,0,i] + patch_size).astype(np.int32)
                
                if peak_illum_idx[1,0,i] - patch_size <= 0:
                    idx_mfirst = 1
                elif peak_illum_idx[1,0,i] + patch_size >= 317:
                    idx_pfirst = 316
                
                # find the illumination index which has max intensity other than zero order
                max_idx_mfirst = np.argmax(max_data[w, :idx_mfirst, i])
                max_idx_pfirst = np.argmax(max_data[w, idx_pfirst:, i]) + idx_pfirst

                # if intensity lower than 0.08 -> invalid
                if max_data[w, max_idx_mfirst, i] < 0.08:
                    max_idx_mfirst = 0
                if max_data[w, max_idx_pfirst, i] < 0.08:
                    max_idx_pfirst = 317
                    
                peak_illum_idx[0,w,i] = max_idx_mfirst
                peak_illum_idx[2,w,i] = max_idx_pfirst
        
        # save peak illumination index data
        np.save(os.path.join(self.npy_dir, 'peak_illum_idx_%s'%self.position), peak_illum_idx)
        
        return peak_illum_idx
        
                
if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
    date = "1007"
    position = "back"
    
    peak_illum_idx = DataProcess(arg, date, position).get_first_idx()