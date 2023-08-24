import numpy as np
import cv2, os,sys
import scipy.io as io
import torch

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
import matplotlib.pyplot as plt
from hyper_sl.utils.ArgParser import Argument
from point_process import PointProcess

class HomographyTransform():
    def __init__(self, arg, date, position):
        # arguments
        self.arg = arg
        self.date = date
        self.wvls = np.array([430, 450, 480, 500, 520, 550, 580, 600, 620, 650, 660])
        self.position = position
        self.total_px = arg.total_px
        
        # directory 
        self.main_dir = "./calibration/dg_calibration_method2/2023%s_data"%self.date
        self.data_dir = os.path.join(self.main_dir, self.position)
        self.processed_data_dir = os.path.join(self.main_dir, "%s_processed"%self.position)
        self.points_dir = self.data_dir + '_points' # detected points dir
        self.points_3d_dir = os.path.join(self.main_dir , "spectralon_depth_%s_%s.npy"%(self.date,self.position)) # spectralon 3d points
        self.pattern_npy_dir = "./calibration/dg_calibration_method2/grid_npy" # pattern npy 
        self.final_dir = './calibration/dg_calibration_method2/2023%s_data/%s_processed_homo'%(date, position)

        # point datas
        self.filter_zero_orders, self.no_filter_zero_orders = self.load_points()
        np.save('./filter_zero_orders.npy', self.filter_zero_orders)
        np.save('./no_filter_zero_orders.npy', self.no_filter_zero_orders)
        
    def get_proj_px(self, dir):
        """
            get (u, v) coords for projected projector sensor plane
        """
        proj_px = np.load(dir)
        
        return proj_px
    
    def get_detected_pts(self, i, proj_px):
        """
            get preprocessed detection points : m, wvl, # points, 3(xyz)
            
            returns detected points
            
        """
        detected_pts_dir = self.points_dir + '/pattern_%04d'%i
        processed_img_dir = self.processed_data_dir + '/pattern_%04d'%i
        detected_pts = PointProcess(self.arg, self.data_dir, detected_pts_dir, processed_img_dir, self.wvls, i, proj_px, self.position).point_process()
        detected_pts = (np.round(detected_pts)).astype(np.int32)
        
        return detected_pts
    
    def load_points(self):
        # projector points
        proj_pts = np.zeros(shape=(self.total_px, 2)) # projector sensor plane pxs : # px, 2

        # 전체 패턴 개수 : 464개의 points / wvl + no filter : 11개 
        filter_zero_orders = np.zeros(shape=(self.total_px, len(self.wvls), 2))
        no_filter_zero_orders = np.zeros(shape=(self.total_px, 2))
        
        for i in range(len(os.listdir(self.pattern_npy_dir))):
            # projector pixel points
            proj_px = self.get_proj_px(os.path.join(self.pattern_npy_dir,"pattern_%05d.npy"%i))
            proj_pts[i] = proj_px
            
            # band pass filter zero orders
            detected_pts = self.get_detected_pts(i, proj_px) # m, wvl, 2 // 패턴의 개수 
            centroids = io.loadmat(os.path.join(self.points_dir, "pattern_%04d"%i, "no_fi_centroid.mat"))['centers']
            
            if len(centroids) == 0:
                no_filter_zero_orders[i] = np.zeros(shape=(2))
            else:
                no_filter_zero_orders[i] = centroids
                
            filter_zero_orders[i] = detected_pts[1].squeeze()
            
        return filter_zero_orders, no_filter_zero_orders

    def transform(self, src_pts, dst_pts):
        
        # Get the matching keypoints
        src_pts = np.float32(src_pts).reshape(-1,1,2)
        dst_pts = np.float32(dst_pts).reshape(-1,1,2)
        
        # get rid of zero points?
        mask_valid = (src_pts > 0.)* (dst_pts > 0.)
        valid_pairs = mask_valid[:, 0, 0]
        
        # Mask the src_pts and dst_pts
        src_pts_masked = src_pts[valid_pairs]
        dst_pts_masked = dst_pts[valid_pairs]
        
        # Step 2: Compute the homography matrix
        H, _ = cv2.findHomography(src_pts_masked, dst_pts_masked, cv2.RANSAC, 5.0)

        return H

    def warp_img(self, image, H):
        height, width = arg.cam_H, arg.cam_W
        warpped_image = cv2.warpPerspective(image, H, (width, height))
        
        return warpped_image
        
    def homography_transformation(self):
        
        # warp perspective for all images & same
        files = os.listdir(self.processed_data_dir)
        
        # make processed homo directory
        if not os.path.exists(self.final_dir):
            os.makedirs(self.final_dir)
        
        for i in range(len(self.wvls)):
            # wvl 마다 detection 된 point load / homography wavelength 마다 구함
            H = self.transform(self.filter_zero_orders[:,i], self.no_filter_zero_orders) # wvl 마다의 homography
            
            for dir in files: # pattern directory
                wvl_dir = os.path.join(self.processed_data_dir, dir)
                wvl_img = cv2.imread(os.path.join(wvl_dir, '%snm.png'%self.wvls[i]))
                
                warpped_img = self.warp_img(wvl_img, H)
                
                new_dir = os.path.join(self.final_dir, dir)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                
                cv2.imwrite(os.path.join(new_dir, '%snm.png'%self.wvls[i]), warpped_img)

        
        
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    bool = False # True : undistort image / False : no undistortion to image
    date = "0822" # date of data
    
    # Homography    
    HomographyTransform(arg, date, position="front").homography_transformation()
    HomographyTransform(arg, date, position="mid")
    HomographyTransform(arg, date, position="back")
        