import cv2, os, sys
import numpy as np

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from scipy.io import loadmat, savemat
from tqdm import tqdm

import define_3dpoints
import define_3dlines
import file_process
from hyper_sl.utils.ArgParser import Argument


def dg_calibration(arg):
    """ file process -> point detection -> dg_calibration code """
    
    # bool = False # True : undistort image / False : no undistortion to image
    # date = "0728" # date of data
    # front = True # True : front spectralon / False : back spectralon
    
    # # file processing : cropping, order datas in pattern and wavelengths
    # file_process.file_process(arg, bool, date, True) # front spectralon
    # file_process.file_process(arg, bool, date, False) # back spectralon
    
    # # wvl
    wvls = np.arange(450, 660, 50)
    pts_num = (arg.proj_H // 10) * (arg.proj_W // 10)
    m_list = arg.m_list
    
    # # find 3d points of front & back spectralon
    # front_world_3d_pts, proj_pts = define_3dpoints.world3d_pts(arg, date, flg=True)
    # back_world_3d_pts, proj_pts = define_3dpoints.world3d_pts(arg, date, flg=False)
    
    # bring saved 3d points
    front_world_3d_pts = np.load('./front_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num -1, 3)
    back_world_3d_pts = np.load('./back_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num -1, 3)
    # bring proj pts
    proj_pts = np.load('./proj_pts.npy')
    
    # reshaping
    a = np.zeros(shape=(arg.m_num, len(wvls), pts_num, 3))
    a[:,:,:2303,:] = front_world_3d_pts
    a[:,:,-1] = front_world_3d_pts[:,:,-1]
    front_world_3d_pts = a
    
    a = np.zeros(shape=(arg.m_num, len(wvls), pts_num, 3))
    a[:,:,:2303,:] = back_world_3d_pts
    a[:,:,-1] = back_world_3d_pts[:,:,-1]
    back_world_3d_pts = a
    
    a = np.zeros(shape=(pts_num, 2))
    a[:2303] = proj_pts
    a[-1] = np.array([[634., 354.]])
    proj_pts = a

    # visualization 3d points of specific order
    # define_3dlines.visualization(front_world_3d_pts, back_world_3d_pts, 2)
    
    front_world_3d_pts_reshape = front_world_3d_pts.reshape(-1, 3)
    back_world_3d_pts_reshape = back_world_3d_pts.reshape(-1, 3)
    
    # delete outliers
    dir_vec_reshape = back_world_3d_pts_reshape - front_world_3d_pts_reshape
    idx = (front_world_3d_pts_reshape[...,2] > 0.) * ( back_world_3d_pts_reshape[...,2] > 0.)
    for i in range(arg.m_num*len(wvls)*pts_num):
        if idx[i] == False:
            dir_vec_reshape[i,:] = 0. # let outlier's direction vector be Zero
    
    dir_vec = dir_vec_reshape.reshape(arg.m_num, len(wvls)*pts_num,3)
    front_world_3d_pts = front_world_3d_pts.reshape(arg.m_num, len(wvls)*pts_num,3)

    # visualization of direction vector of specific order
    # define_3dlines.dir_visualization(dir_vec, front_world_3d_pts, 2)

    dir_vec = dir_vec_reshape.reshape(arg.m_num, len(wvls), pts_num, 3)
    front_world_3d_pts = front_world_3d_pts.reshape(arg.m_num, len(wvls), pts_num, 3)

    # depth = np.arange(0.6, 0.9, 0.001)
    depth = np.arange(0.6, 0.9, 0.01)
    
    dat_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/dat/method2"

    for z in depth:
        t = (z - front_world_3d_pts[...,2]) / dir_vec[...,2]
        pts = front_world_3d_pts + dir_vec * t[:,:,:,np.newaxis]
        
        for m in tqdm(range(arg.m_num)):
            for w in tqdm(range(len(wvls)), leave = False):
                
                pts_x = pts[m, w, :, 0]
                pts_y = pts[m, w, :, 1]
                pts_z = pts[m, w, :, 2]
                
                gt_x = proj_pts[:,0]
                gt_y = proj_pts[:,1]
                
                savemat(os.path.join(dat_path, 'dispersion_coordinates_m%d_wvl%d_depth%02dcm.mat'%(m_list[m], wvls[w], z *100)), {'x': pts_x, 'y': pts_y, 'z': pts_z, 'xo': gt_x, 'yo': gt_y})
                # savemat(os.path.join(dat_path, 'dispersion_coordinates_m%d_wvl%d_depth%03dmm.mat'%(m_list[m], wvls[w], z *1000)), {'x': pts_x, 'y': pts_y, 'z': pts_z, 'xo': gt_x, 'yo': gt_y})

if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()

    dg_calibration(arg)

    print('end')