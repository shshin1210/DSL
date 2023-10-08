import numpy as np
import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument

"""
    Create HDR image (which creates PPG graph)
"""

class HDR():
    def __init__(self, arg):
        
        # args
        self.invalid_intensity_ratio = arg.invalid_intensity_ratio
        self.max_intensity = 2**16
        self.n_illum = arg.illum_num
        
        # idx min max for exposure and intensity normalization
        self.idx_minmax = -1
        
        # exposure 
        self.ex_time = np.array([arg.exp_min, arg.exp_max])
        self.ex_min = self.ex_time[self.idx_minmax]
        self.exposure = self.ex_time / self.ex_min
        
        # intensity
        self.intensity = np.array([arg.intensity_min, arg.intensity_max])
        self.intensity_normalization_pts = np.array(arg.intensity_normalization_pts)
        self.p_size = 10
        
        # dir
        self.real_data_dir = os.path.join(arg.real_data_dir, '2023%s_real_data'%arg.real_data_date)
        
        
    def safe_subtract(self, a,b):
        """
            safe subtraction for uint16
        """
        difference = np.where(a>b, a-b, 0)
        difference = np.clip(np.round(difference), 0, 65535).astype(np.uint16)    
        
        return difference.astype(a.dtype)
    
    def cal_radiance_weight(self):
        """
            calculate radiance weight for different intensity illuminations
            
            returns : radiance_weight
        
        """
        # calculate radiance_weight
        exp_img_path = os.path.join(self.real_data_dir, 'intensity_%d_white_crop/calibration00/capture_%04d.png')
        exp_img_black_path = os.path.join(self.real_data_dir, 'step2_%sms_black_crop/calibration00/capture_%04d.png')

        exp_images = np.array([cv2.imread(exp_img_path%(self.intensity[k]*100, 0), -1)[:,:,::-1] for k in range(len(self.intensity))])
        exp_black_images = np.array([cv2.imread(exp_img_black_path%(160, 0), -1)[:,:,::-1] for k in self.ex_time])

        # remove black image
        exp_images_bgrm = self.safe_subtract(exp_images, exp_black_images)
        
        radiance_values = np.zeros(shape=(len(exp_images_bgrm), 3))
                
        for i in range(len(exp_images_bgrm)):
            y_idx_start, y_idx_end = (self.intensity_normalization_pts[1] - self.p_size//2).astype(np.int32), (self.intensity_normalization_pts[1] + self.p_size//2).astype(np.int32)
            x_idx_start, x_idx_end = (self.intensity_normalization_pts[0] - self.p_size//2).astype(np.int32), (self.intensity_normalization_pts[0] + self.p_size//2).astype(np.int32)
            
            radiance_values[i] = exp_images_bgrm[i][y_idx_start:y_idx_end, x_idx_start:x_idx_end].mean(axis = (0,1))
        
        radiance_weight = (radiance_values / radiance_values[self.idx_minmax])
        
        return radiance_weight
    
    def weight_trapezoid(self): 
        
        # weight trapezoid for original image
        weight_trapezoid = np.zeros(self.max_intensity)
        intv = float(self.max_intensity) * self.invalid_intensity_ratio

        for i in range(self.max_intensity):
            if i < intv:
                weight_trapezoid[i] = 0  
            elif i < intv * 2:
                weight_trapezoid[i] = (i - intv) / intv
            elif i < self.max_intensity - (intv * 2):
                weight_trapezoid[i] = 1
            elif i < self.max_intensity - intv:
                weight_trapezoid[i] = (self.max_intensity - intv - i) / intv
            else:
                weight_trapezoid[i] = 0
        
        # weight trapezoid for bgrm image
        weight_trapezoid_bgrm = np.zeros(self.max_intensity)
        intv = float(self.max_intensity) * self.invalid_intensity_ratio

        for i in range(self.max_intensity):
            if i < intv: # 3 
                weight_trapezoid_bgrm[i] = 0  
            elif i < intv * 2:
                weight_trapezoid_bgrm[i] = (i - intv) / intv
            else:
                weight_trapezoid_bgrm[i] = 1
                
        return weight_trapezoid, weight_trapezoid_bgrm
    
    def hdr(self, ldr_images, ldr_images_bgrm):
        
        """
            make hdr image
            
        """
        weight_trapezoid, weight_trapezoid_bgrm = self.weight_trapezoid()
        radiance_weight = self.cal_radiance_weight()
        
        weighted_images = [weight_trapezoid[image] for image in ldr_images]
        weighted_images_bgrm = [weight_trapezoid_bgrm[image] for image in ldr_images_bgrm]
        
        # take the minimum weight 
        weighted_images_final = np.minimum(weighted_images, weighted_images_bgrm)
        
        # exposure normalization
        radiance_images = [np.multiply(weighted_images_final[i], ldr_images_bgrm[i] / (radiance_weight[i] * self.exposure[i])) for i in range(len(ldr_images_bgrm))]

        # intensity normalization
        weight_sum_image = np.sum(weighted_images_final, axis=0)
        radiance_sum_image = np.sum(radiance_images, axis=0)

        idx_invalid = (weight_sum_image == 0)
        weight_sum_image[idx_invalid] = 1
        radiance_sum_image[idx_invalid] = 0    
        
        return np.divide(radiance_sum_image, weight_sum_image), idx_invalid, weight_sum_image
    
    
    def make_hdr(self):
        """
            create hdr image with two different exposure time and intensity level
        """
        hdr_imgs = []

        # erase black image and get hdr image
        for i in range(self.n_illum):    
            ldr_path = os.path.join(self.real_data_dir, 'step2_%sms_crop/calibration00/capture_%04d.png')
            black_path = os.path.join(self.real_data_dir, 'step2_%sms_black_crop/calibration00/capture_%04d.png')

            ldr_images = np.array([cv2.imread(ldr_path%(k, i), -1)[:,:,::-1] for k in self.ex_time])
            ldr_black_images = np.array([cv2.imread(black_path%(k, 0), -1)[:,:,::-1] for k in self.ex_time])
            
            # print(ldr_images.max(), ldr_images.min(), ldr_black_images.max(), ldr_black_images.min())    
            
            ldr_images_bgrm = np.clip(self.safe_subtract(ldr_images, ldr_black_images), 0., 2**16)
            ldr_images_bgrm = ldr_images_bgrm.astype(np.uint16)
            
            # print(final_ldr_images.max(), final_ldr_images.min())
                
            hdr_img, invalid_map, weight_map = self.hdr(ldr_images, ldr_images_bgrm)
            hdr_imgs.append(hdr_img)

            if i % 10 == 0:
                print('%03d-th finished'%i)

        hdr_imgs = np.array(hdr_imgs)
        
        return hdr_imgs
    
    
if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
    hdr_imgs = HDR(arg).make_hdr()
