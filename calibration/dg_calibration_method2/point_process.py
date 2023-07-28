import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
from scipy.io import loadmat

def point_process(arg, data_dir, detected_pts_dir, wvls, n_patt):

    processed_points = np.zeros(shape=(arg.m_num, len(wvls), 1, 2))
    
    for w in range(len(wvls)):
        wvl_point = np.array(loadmat(os.path.join(detected_pts_dir,'%dnm_centroid.mat' %(wvls[w])))['centers'])            
        wvl_point -= 1
        # wvl_point = np.array([pts[0][0][0] for pts in wvl_point])

        zero, first_m0, first_m2 = find_order(arg, data_dir, wvl_point, wvls[w], n_patt)
        
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
    
def find_order(arg, data_dir, wvl_point, wvl, n_patt):
    """
        Find which order each points belong to
        orders : -1, 0, 1
        
        grid_pts : number of white grid pattern for zero-order
        wvl_point : coordinate points of all grid points detected by dot_detection.m
        wvl : 450 - 650 nm, 50 nm interval
        n_patt : number of patterns

    """
    
    img = cv2.imread(data_dir + '_processed/pattern_%04d/%dnm.png'%(n_patt, wvl))
    img_m = np.mean(img, axis = 2)
    
    # 아무것도 안찍힌 데이터 처리
    if wvl_point.shape[0] == 0:
        wvl_point = np.zeros(shape=(arg.m_num,2))
    
    # sort with x-axis to find avg of each cols
    sorted_idx = np.argsort(wvl_point[...,0], axis = 0)
    x_sort = np.take_along_axis(wvl_point[...,0], sorted_idx, axis = 0)
    y_sort = np.take_along_axis(wvl_point[...,1], sorted_idx, axis = 0)
    wvl_point[...,0], wvl_point[...,1] = x_sort, y_sort

    # Outlier 처리
    
    
    # intensity for all points
    pts = np.array([img_m[i[1].astype(np.int16) ,i[0].astype(np.int16)] for i in wvl_point])

    # grid pts 개수
    grid_pts = 1

    # separate with # of grid point
    avg_t = np.average(pts[:grid_pts])
    avg_b = np.average(pts[grid_pts:])
    
    #### Zero 만 있는지, zero & first 다 있는지, 
    # zero 만 있는 경우
    zero_arr = np.zeros(shape=(grid_pts, 2))
    
    # m = 0, m = 1 x 좌표 기준
    x_max = 520
    
    # 하나의 1st order만 있는경우
    if wvl_point.shape[0] == grid_pts : 
        if x_max < wvl_point[0,0]: # x 좌표 비교
            first_m2 = wvl_point
            return zero_arr, zero_arr, first_m2
        
        if x_max > wvl_point[0,0]:
            first_m0 = wvl_point
            return zero_arr, first_m0, zero_arr

        else:
            zero = wvl_point
            return zero, zero_arr, zero_arr
    
    # 두개의 m order가 있는 경우
    elif wvl_point.shape[0] == grid_pts *2:
        # separate first and zero
        if avg_t < avg_b:
            first = wvl_point[:grid_pts]
            zero = wvl_point[grid_pts:]
        else:
            zero = wvl_point[:grid_pts]
            first = wvl_point[grid_pts:]
            
        # choose between order m = -1, 1
        if zero[:,0].mean() > first[:,0].mean():
            first_m2 = first
            return zero, zero_arr, first_m2
        else:
            first_m0 = first
            return zero, first_m0, zero_arr
    
    # 세개의 1st order가 있는 경우
    else:
        first_m2 = wvl_point[:grid_pts]
        zero = wvl_point[grid_pts:grid_pts*2]
        first_m0 = wvl_point[grid_pts*2:]
        
        return zero, first_m0, first_m2

if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    date = './calibration/dg_calibration_method2/20230728_data/front'
    point_dir = date + '_points'
    
    N_pattern = len(os.listdir(point_dir))
    # wvls = np.arange(450, 660, 50)
    wvls = np.array([450])
    
    for i in range(N_pattern):
        detected_pts_dir = point_dir + '/pattern_%04d'%i
        processed_points = point_process(arg, date, detected_pts_dir, wvls, i)

        print('end')