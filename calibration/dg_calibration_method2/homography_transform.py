import numpy as np
import cv2, os,sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
import matplotlib.pyplot as plt
from hyper_sl.utils.ArgParser import Argument


class HomographyTransform():
    def __init__(self, arg, date, position):
        # arguments
        self.arg = arg
        self.wvls = np.array([430, 450, 480, 500, 520, 550, 580, 600, 620, 650, 660])

        # directory
        self.processed_data_dir = './calibration/dg_calibration_method2/2023%s_data/%s_processed'%(date, position)
        self.homography_data_dir = './calibration/dg_calibration_method2/2023%s_data/homography'%date
        self.final_dir = './calibration/dg_calibration_method2/2023%s_data/%s_processed_homo'%(date, position)
        
    def load_img(self, type):
        """
            load homography images
            type : band pass filter holder / wheel / with no filter
            
        """
        
        if type == "wheel":
            image = cv2.imread(os.path.join(self.homography_data_dir, 'wheel.png'), cv2.IMREAD_GRAYSCALE)
        elif type == "holder":
            image = cv2.imread(os.path.join(self.homography_data_dir, 'holder.png'), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(os.path.join(self.homography_data_dir, 'plain.png'), cv2.IMREAD_GRAYSCALE)
        
        return image
    
    def detect_compute(self, image1, image2):
        # Step 1: Detect and match features using ORB detector as an example
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        return keypoints1, descriptors1, keypoints2, descriptors2, matches
    
    def transform(self, image1, image2):
        
        # Parameters for the checkerboard (inner corners)
        rows = 6 # Example value
        cols = 9 # Example value

        # Step 1: Detect checkerboard corners
        ret1, corners1 = cv2.findChessboardCorners(image1, (rows, cols))
        ret2, corners2 = cv2.findChessboardCorners(image2, (rows, cols))

        if ret1 and ret2:
            # Optional: Refine the corner detections
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(image1, corners1, (11,11), (-1,-1), criteria)
            cv2.cornerSubPix(image2, corners2, (11,11), (-1,-1), criteria)

        # # Step 1: Detect and match features using ORB detector as an example
        # keypoints1, descriptors1, keypoints2, descriptors2, matches = self.detect_compute(image1, image2)
        
        # # Sort matches based on their distances
        # matches = sorted(matches, key = lambda x:x.distance)

        # # Take the top N matches
        # N = 50
        # best_matches = matches[:N]

        # # Get the matching keypoints
        # src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in best_matches ]).reshape(-1,1,2)
        # dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in best_matches ]).reshape(-1,1,2)


        # Step 2: Compute the homography matrix
        # H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        H, _ = cv2.findHomography(corners1, corners2, cv2.RANSAC, 5.0)


        return H
    
    def warp_img(self, image, H):
        height, width = arg.cam_H, arg.cam_W
        warpped_image = cv2.warpPerspective(image, H, (width, height))
        
        return warpped_image
        
    def homography_transformation(self):
        # Load homography images
        wheel_img = self.load_img("wheel")
        holder_img = self.load_img("holder")
        plain_img = self.load_img("plain_img")
        
        # Homography 
        # wheel_to_plain_H = self.transform(wheel_img, plain_img)
        wheel_to_plain_H = self.transform(plain_img, wheel_img)

        holder_to_plain_H = self.transform(plain_img, holder_img)
        
        # warp perspective for all images & same
        # wheel : 430nm ~ 550nm
        # holder : 580nm ~ 660nm
        files = os.listdir(self.processed_data_dir)
        # make dir
        if not os.path.exists(self.final_dir):
            os.makedirs(self.final_dir)
        
        for i in range(len(self.wvls)):
            for dir in files:
                if i < 6:
                    wvl_dir = os.path.join(self.processed_data_dir, dir)
                    wvl_img = cv2.imread(os.path.join(wvl_dir, '%snm.png'%self.wvls[i]))
                    
                    warpped_img = self.warp_img(wvl_img, wheel_to_plain_H)
                    
                    new_dir = os.path.join(self.final_dir, dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    
                    cv2.imwrite(os.path.join(new_dir, '%snm.png'%self.wvls[i]), warpped_img)
                    
                else:
                    wvl_dir = os.path.join(self.processed_data_dir, dir)
                    wvl_img = cv2.imread(os.path.join(wvl_dir, '%snm.png'%self.wvls[i]))
                    
                    warpped_img = self.warp_img(wvl_img, holder_to_plain_H)
                    
                    new_dir = os.path.join(self.final_dir, dir)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    
                    cv2.imwrite(os.path.join(new_dir, '%snm.png'%self.wvls[i]), warpped_img)
            
        
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    bool = False # True : undistort image / False : no undistortion to image
    date = "0821" # date of data
    
    # Homography    
    HomographyTransform(arg, date, position="front").homography_transformation()
    HomographyTransform(arg, date, position="mid")
    HomographyTransform(arg, date, position="back")
        