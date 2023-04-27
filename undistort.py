import cv2, os
import numpy as np
from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import data_process
from hyper_sl.utils import calibrated_params
import matplotlib.pyplot as plt

def illum_imgs(path, illum_num, H, W, type):
    """
        Returns images for illumination 40 patterns
    """
    
    # illum image
    img_list = np.zeros(shape = (H, W, illum_num, 3))
    for i in range(illum_num):
        if type == "pattern":
            # illumination pattern
            img = cv2.imread(os.path.join(path, "pattern_%02d.png"%i), -1)/65535.
        else:
            # cam captured image
            img = cv2.imread(os.path.join(path, 'capture_%04d.png'%i), -1)/65535.
            img = data_process.crop(img)
            
        img_list[:,:,i] = img

    return img_list

def undistort(img, int_ext, dist_ext, H, W, illum_num):
    img_undist = np.zeros(shape=(H, W, illum_num, 3))
    for i in range(illum_num):
        img_undist[:,:,i] = cv2.undistort(img[:,:,i], int_ext, dist_ext)
    
    return img_undist

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

    cam_path = "/home/shshin/Scalable-Hyperspectral-3D-Imaging/dataset/data/real_data/scene0000"
    patt_path = "/home/shshin/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/illum/graycode_pattern"
    
    cam_imgs = illum_imgs(cam_path, arg.illum_num, arg.cam_H, arg.cam_W, "cam")
    patt_imgs = illum_imgs(patt_path, arg.illum_num, arg.proj_H, arg.proj_W, "pattern")
    
    cam_undistort = undistort(cam_imgs, cam_int, cam_dist, arg.cam_H, arg.cam_W, arg.illum_num)
    illum_undistort = undistort(patt_imgs, proj_int, proj_dist, arg.proj_H, arg.proj_W, arg.illum_num)
    
    vis(cam_undistort)
    vis(illum_undistort)
    
    print('end')