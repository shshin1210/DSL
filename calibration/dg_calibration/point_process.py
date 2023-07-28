import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
from scipy.io import loadmat

def point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, n_patt):

    pixel_num = grid_pts
    processed_points = np.zeros(shape=(arg.m_num, len(wvls), pixel_num, 2))
    
    for w in range(len(wvls)):
        wvl_point = np.array(loadmat(os.path.join(pattern_dir,'%dnm_centroid.mat' %(wvls[w])))['centers'])            
        wvl_point -= 1
        # wvl_point = np.array([pts[0][0][0] for pts in wvl_point])

        zero, first_m0, first_m2 = find_order(grid_pts, total_dir, date, wvl_point, wvls[w], n_patt)
        
        # zero order
        processed_points[1] = zero[np.newaxis,:,:]
        processed_points[0, w, :, :] = first_m0
        processed_points[2, w, :, :] = first_m2
    
    # sorting points
    sorted_idx = np.argsort(-processed_points[...,1], axis = 2)
    x_sort = np.take_along_axis(processed_points[...,0], sorted_idx, axis = 2)
    y_sort = np.take_along_axis(processed_points[...,1], sorted_idx, axis = 2)
    
    processed_points[...,0], processed_points[...,1] = x_sort, y_sort
    
    return processed_points
    
def find_order(grid_pts, total_dir, date, wvl_point, wvl, n_patt):
    """
        Find which order each points belong to
        orders : -1, 0, 1
        
        grid_pts : number of white grid pattern for zero-order
        total_dir : directory path
        date : date of data directory
        wvl_point : coordinate points of all grid points detected by dot_detection.m
        wvl : 450 - 650 nm, 50 nm interval
        n_patt : number of patterns

    """
    dir = total_dir + date + '_processed/'
    img = cv2.imread(dir+ 'pattern_%02d/%03dnm.png'%(n_patt, wvl))
    img_m = img.mean(axis = 2)
    
    # sort with x-axis to find avg of each cols
    sorted_idx = np.argsort(wvl_point[...,0], axis = 0)
    x_sort = np.take_along_axis(wvl_point[...,0], sorted_idx, axis = 0)
    y_sort = np.take_along_axis(wvl_point[...,1], sorted_idx, axis = 0)
    wvl_point[...,0], wvl_point[...,1] = x_sort, y_sort

    # intensity for all points
    pts = np.array([img_m[i[1].astype(np.int16) ,i[0].astype(np.int16)] for i in wvl_point])

    # separate with # of grid point
    avg_t = np.average(pts[:grid_pts])
    avg_b = np.average(pts[grid_pts:])
    
    #### Zero 만 있는지, zero & first 다 있는지, 
    # zero 만 있는 경우
    zero_arr = np.zeros(shape=(grid_pts, 2))
    
    # 하나의 1st order만 있는경우
    if wvl_point.shape[0] == grid_pts : 
        zero = wvl_point
        return zero, zero_arr, zero_arr
    
    elif wvl_point.shape[0] == grid_pts *2:
        # separate first and zero
        if avg_t < avg_b:
            first = wvl_point[:grid_pts]
            zero = wvl_point[grid_pts:]
        else:
            zero = wvl_point[:grid_pts]
            first = wvl_point[grid_pts:]
            
        # split order m = -1, 1
        if zero[:,0].mean() > first[:,0].mean():
            first_m2 = first
            return zero, zero_arr, first_m2
        else:
            first_m0 = first
            return zero, first_m0, zero_arr
    
    # 두개의 1st order가 있는 경우
    else:
        first_m2 = wvl_point[:grid_pts]
        zero = wvl_point[grid_pts:grid_pts*2]
        first_m0 = wvl_point[grid_pts*2:]
        
        return zero, first_m0, first_m2

if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    total_dir = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/"
    date = 'test_2023_07_09_15_37_(2)'
    point_dir = total_dir + date + '_points'
    
    N_pattern = len(os.listdir(point_dir))
    wvls = np.arange(450, 660, 50)
    grid_pts = 5
    
    for i in range(N_pattern):
        pattern_dir = point_dir + '/pattern_%02d'%i
        processed_points = point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, i)

        print('end')