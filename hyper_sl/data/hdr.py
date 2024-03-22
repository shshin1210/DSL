import numpy as np
import cv2, os, sys
from hyper_sl.utils.ArgParser import Argument

"""
    Create HDR image (which creates PPG graph)
    
"""

class HDR():
    """
        Reconstruct hyperspectral iamge

        Arguments
            - invalid_intensity_ratio: height and width of camera
            - n_illum: number of white scan line pattern
            
            Exposure
            - ex_time: exposure times (min and max exposure time)

            Intensity
            - intensity: intensity of projected patterns
            - p_size: patch size of black patch

    """
    def __init__(self, arg):
        
        # args
        self.invalid_intensity_ratio = arg.invalid_intensity_ratio
        self.n_illum = arg.illum_num
        self.max_intensity = arg.max_intensity
        
        # exposure 
        self.ex_time = np.array([arg.exp_min, arg.exp_max])
        self.exposure = self.ex_time / arg.exp_max
        
        # intensity
        self.intensity = np.array([arg.intensity_min, arg.intensity_max])
        self.intensity_normalization_pts = np.array(arg.intensity_normalization_pts)
        self.p_size = 10
        
        # directoy
        self.path_to_intensity1 = arg.path_to_intensity1
        self.path_to_intensity2 = arg.path_to_intensity2
        
        self.path_to_black_exp1 = arg.path_to_black_exp1
        self.path_to_black_exp2 = arg.path_to_black_exp2
        
        self.path_to_ldr_exp1 = arg.path_to_ldr_exp1
        self.path_to_ldr_exp2 = arg.path_to_ldr_exp2
        
        
    def safe_subtract(self, a,b):
        """
            safe subtraction for uint16
        """
        difference = np.where(a>b, a-b, 0)
        difference = np.clip(np.round(difference), 0, 65535).astype(np.uint16)    
        
        return difference.astype(a.dtype)
    
    def cal_radiance_weight(self):
        """
            calculate radiance weight for different intensity illuminations normalization using colorchecker's black patch
            
            returns : radiance_weight
        
        """
        # calculate radiance_weight
        exp_img_path = np.array([self.path_to_intensity1, self.path_to_intensity2])
        exp_img_black_path = np.array([self.path_to_black_exp1, self.path_to_black_exp2])

        exp_images = np.array([cv2.imread(exp_img_path[k], -1)[:,:,::-1] for k in range(len(self.intensity))])
        exp_black_images = np.array([cv2.imread(exp_img_black_path[k], -1)[:,:,::-1] for k in self.ex_time])

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
            if i < intv:
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
            ldr_path = np.array([os.listdir(self.path_to_ldr_exp1), os.listdir(self.path_to_ldr_exp2)])
            black_path = np.array([os.listdir(self.path_to_black_exp1), os.listdir(self.path_to_black_exp2)])

            ldr_images = np.array([cv2.imread(ldr_path[k, i], -1)[:,:,::-1] for k in self.ex_time])
            ldr_black_images = np.array([cv2.imread(black_path[k, 0], -1)[:,:,::-1] for k in self.ex_time])
                        
            ldr_images_bgrm = np.clip(self.safe_subtract(ldr_images, ldr_black_images), 0., self.max_intensity)
            ldr_images_bgrm = ldr_images_bgrm.astype(np.uint16)
                            
            hdr_img, invalid_map, weight_map = self.hdr(ldr_images, ldr_images_bgrm)
            hdr_imgs.append(hdr_img)

        hdr_imgs = np.array(hdr_imgs)
        
        return hdr_imgs
    
    
if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
    hdr_imgs = HDR(arg).make_hdr()
