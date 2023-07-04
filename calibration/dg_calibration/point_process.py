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
        # wvl_point = np.array([pts[0][0][0] for pts in wvl_point])
        
        zero, first, bool = find_order(grid_pts, total_dir, date, wvl_point, wvls[w], n_patt)
        
        # zero order
        processed_points[1] = zero[np.newaxis,:,:]
        
        if bool == False:
            processed_points[0, w, :, :] = first
        else:
            processed_points[2, w, :, :] = first
    
    # sorting points
    sorted_idx = np.argsort(-processed_points[...,1], axis = 2)
    x_sort = np.take_along_axis(processed_points[...,0], sorted_idx, axis = 2)
    y_sort = np.take_along_axis(processed_points[...,1], sorted_idx, axis = 2)
    
    processed_points[...,0], processed_points[...,1] = x_sort, y_sort
    
    return processed_points
    
def find_order(grid_pts, total_dir, date, wvl_point, wvl, n_patt):
    dir = total_dir + date + '_processed/'
    img = cv2.imread(dir+ 'pattern_%02d/%03dnm.png'%(n_patt, wvl))
    img_m = img.mean(axis = 2)
    
    sorted_idx = np.argsort(wvl_point[...,0], axis = 0)
    x_sort = np.take_along_axis(wvl_point[...,0], sorted_idx, axis = 0)
    y_sort = np.take_along_axis(wvl_point[...,1], sorted_idx, axis = 0)
    
    wvl_point[...,0], wvl_point[...,1] = x_sort, y_sort

    
    pts = np.array([img_m[i[1].astype(np.int16) ,i[0].astype(np.int16)] for i in wvl_point])

    avg_t = np.average(pts[:grid_pts])
    avg_b = np.average(pts[grid_pts:])
    
    if avg_t < avg_b:
        first = wvl_point[:grid_pts]
        zero = wvl_point[grid_pts:]
    else:
        zero = wvl_point[:grid_pts]
        first = wvl_point[grid_pts:]
        
    # split order m = -1, 1
    # m = -1 order / m = 1 order
    if zero[:,0].mean() > first[:,0].mean():
        first_m2 = first
        return zero, first_m2, True
    else:
        first_m0 = first
        return zero, first_m0, False

if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    total_dir = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/"
    date = 'test_2023_07_03_17_15'
    point_dir = total_dir + date + '_points'
    
    N_pattern = len(os.listdir(point_dir))
    wvls = np.arange(450, 660, 50)
    grid_pts = 4
    
    for i in range(N_pattern):
        pattern_dir = point_dir + '/pattern_%02d'%i
        processed_points = point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, i)

        print('end')