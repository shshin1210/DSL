import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

import numpy as np
from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import data_process
from hyper_sl.utils import calibrated_params
import matplotlib.pyplot as plt

def undistort(img_dir, int_ext, dist_ext):
    files = os.listdir(img_dir)
    for idx, fn in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, fn), -1) / 255.
        cv2.imwrite(os.path.join(img_dir, fn[:-4] + '_undist.png'), cv2.undistort(img, int_ext, dist_ext)*255.)        
    

def vis(data):
    illum_num = 40
    max_images_per_column = 5
    num_columns = (illum_num + max_images_per_column - 1) // max_images_per_column
    plt.figure(figsize=(10, 3*num_columns))

    for c in range(num_columns):
        start_index = c * max_images_per_column
        end_index = min(start_index + max_images_per_column, illum_num)
        num_images = end_index - start_index
                
        for i in range(num_images):
            plt.subplot(num_columns, num_images, i + c * num_images + 1)
            plt.imshow(data[:, :, i + start_index], vmin=0., vmax=1.)
            plt.axis('off')
            plt.title(f"Image {i + start_index}")
            # cv2.imwrite('%04d_img.png'%(i+start_index), data[:, :, i + start_index, ::-1]*255.)
                    
            if i + start_index == illum_num - 1:
                plt.colorbar()
    plt.show()

if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    # bring calibration parameters
    cam_int, cam_dist = calibrated_params.bring_params(arg.calibration_param_path, "cam")
    proj_int, proj_dist, _, _ = calibrated_params.bring_params(arg.calibration_param_path, "proj")

    cam_path = "../../calibration/diff_captured_img_0503"
    
    undistort(cam_path, cam_int, cam_dist)

    patt_path = "/home/shshin/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/illum/graycode_pattern"

    print('end')