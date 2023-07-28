import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import calibrated_params
import numpy as np

"""
    Crop and undistort camera captured images for each grid patterns and wavelengths
    
    input : captured test_date_file directory
    output : creates directory for each wavelength and grid pattern with cropped and undistorted image

"""

def crop(arg, data):
    """
        input : image to crop
        output : returns a cropped single image
    """
    start_x, start_y = 75,77
    res_x, res_y = arg.cam_W, arg.cam_H
    data = data[start_y: start_y + res_y, start_x:start_x+res_x]
    
    return data

def undistort(img_dir, int_mtx, dist_mtx, bit, bool):
    """
        input : image directory(where image is located)
                intrinsic, extrinsic matrix, bit
                
        output : returns single undistorted image(in image_dir)
    """

    img = cv2.imread(os.path.join(img_dir), -1) / bit
    
    if bool == True:
        img_undistort = cv2.undistort(img, int_mtx, dist_mtx)*255.
    else: 
        img_undistort = img * 255.
    
    return img_undistort

def save_img(save_dir, img, fn):
    """
        input : directory to save an image, image file to save, file name
        output : saves in save_dir
    """
    cv2.imwrite(os.path.join(save_dir, fn), img)

def main(arg, bool):
    """
        capture images with 650 ~ 450 nm sequence in calibration0~4 folder
        
        **test_fn = directory name
        
        **True : undistort image
          False : no undistortion to image
    """
    
    wvls = np.arange(450, 660, 50)
    
    cam_int, cam_dist = calibrated_params.bring_params(arg.calibration_param_path, "cam")
    
    test_fn = "test_2023_07_09_15_37_(2)"
    img_test_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/" + test_fn
    
    # files for each wvls
    files = os.listdir(img_test_path)
    
    for idx, file in enumerate(files):
        # files for each grid patterns
        inner_files = os.listdir(os.path.join(img_test_path,file))
        
        for idx2, inner_file in enumerate(inner_files):
            img_path = os.path.join(img_test_path, file, inner_file)
            img_undistort = undistort(img_path, cam_int, cam_dist, 65535., bool)
            img_undistort_crop = crop(arg, img_undistort)
            
            # where to save cropped and undistorted image
            save_dir = os.path.join('%s'%(img_test_path + "_processed"), 'pattern_%02d'%idx2)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_img(save_dir, img_undistort_crop, '%dnm.png'%wvls[idx])


if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()

    bool = False # True : undistort image / False : no undistortion to image
    main(arg, bool)

    print('end')