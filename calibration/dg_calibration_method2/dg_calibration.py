import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import calibrated_params
import numpy as np
import point_process
import matplotlib.pyplot as plt

def get_3d_points(points_3d_dir):
    points_3d = np.load(points_3d_dir)
    
    return points_3d

def get_proj_px(pattern_npy_dir):
    proj_px = np.load(pattern_npy_dir)
    
    return proj_px

def get_detected_pts(arg, point_dir, data_dir, wvls, i):
    detected_pts_dir = point_dir + '/pattern_%04d'%i
    detected_pts = point_process.point_process(arg, data_dir, detected_pts_dir, wvls, i)
    detected_pts = (np.round(detected_pts)).astype(np.int32)
    
    return detected_pts

def visualization(world_3d_pts):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    
    ax.scatter(world_3d_pts[:,:,:,0].flatten(), world_3d_pts[:,:,:,1].flatten(), world_3d_pts[:,:,:,2].flatten())

    ax.set_xlim([-0.05,0.05])
    ax.set_ylim([-0.05,0.05])
    ax.set_zlim([-0.05,0.03])

    plt.show()
    
def main(arg, flg):
    # Variable Date
    date = "0728"    
    front = flg
    
    wvls = np.arange(450, 660, 50)
    # wvls = np.array([450])
    wvls_num = len(wvls)
    total_px = (arg.proj_H//10)*(arg.proj_W//10)
    
    if front == True:
        position = "front"
    else:
        position = "back"
    
    # directory
    main_dir = "./calibration/dg_calibration_method2/2023%s_data"%date
    data_dir = os.path.join(main_dir, position)
    
    # detected points dir
    point_dir = data_dir + '_points'
    # spectralon 3d points
    points_3d_dir = os.path.join(main_dir , "spectralon_depth_%s_%s.npy"%(date,position))
    # pattern npy 
    pattern_npy_dir = "./calibration/dg_calibration_method2/grid_npy"

    # 3d points
    points_3d = get_3d_points(points_3d_dir)
    
    # New arrays : m, wvl, # px(=1), 2
    world_3d_pts = np.zeros(shape=(arg.m_num, wvls_num, total_px, 3))
    world_3d_pts_reshape = world_3d_pts.reshape(-1, total_px, 3) # m * wvl, # px, 3
    # projector sensor plane pxs : #px, 2
    proj_pts = np.zeros(shape=(total_px, 2))
    
    for i in range(len(os.listdir(pattern_npy_dir))-1):
        # proj pixel center points
        pattern_dir = os.path.join(pattern_npy_dir, "pattern_%05d.npy"%i)
        
        # projector pixel points
        proj_px = get_proj_px(pattern_dir)
        proj_pts[i] = proj_px
        
        # detected pts
        detected_pts = get_detected_pts(arg, point_dir, data_dir, wvls, i) # m, wvl, 2
        detected_pts_reshape = detected_pts.reshape(-1, 2) # (x, y 순)
        
        world_3d_pts_reshape[:,i,:] = points_3d[detected_pts_reshape[:,1], detected_pts_reshape[:,0]]

    return world_3d_pts_reshape, proj_pts


if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()

    flg = True # True : front spectralon / False : back spectralon
    
    front_world_3d_pts_reshape, proj_pts = main(arg, flg=True)
    back_world_3d_pts_reshape, proj_pts = main(arg, flg=False)
