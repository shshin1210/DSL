import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
from scipy.io import loadmat

def point_process(arg, data_dir, detected_pts_dir, wvls, n_patt, proj_px, position):

    prev_wvl_point = wvl_point = np.array(loadmat(os.path.join(detected_pts_dir,'%dnm_centroid.mat' %(wvls[0])))['centers'])            
    processed_points = np.zeros(shape=(arg.m_num, len(wvls), 1, 2))
    
    for w in range(len(wvls)):
        wvl_point = np.array(loadmat(os.path.join(detected_pts_dir,'%dnm_centroid.mat' %(wvls[w])))['centers'])            
        wvl_point -= 1
        
        zero, first_m0, first_m2, prev_wvl_point = find_order(arg, data_dir, wvl_point, wvls[w], n_patt, prev_wvl_point, proj_px, position)
        
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

def one_order(wvl_point,zero_arr, proj_px, position):
    if position == "front": x_max, x_min = 500, 10
    elif position == "mid": x_max, x_min = 450, 30
    else: x_max, x_min = 450, 30 # ===================================================== back spectralon 수정 =====================================================
        
    if x_max < proj_px[0]: # x 좌표 비교
        first_m2 = wvl_point
        return zero_arr, zero_arr, first_m2
    
    elif x_min > proj_px[0]:
        first_m0 = wvl_point
        return zero_arr, first_m0, zero_arr

    else:
        zero = wvl_point
        return zero, zero_arr, zero_arr

def two_order(avg_t, avg_b, wvl_point, grid_pts, zero_arr):
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
    
def three_order(wvl_point, grid_pts):
    first_m2 = wvl_point[:grid_pts]
    zero = wvl_point[grid_pts:grid_pts*2]
    first_m0 = wvl_point[grid_pts*2:]
    
    return zero, first_m0, first_m2

def out_lier(wvl_point, prev_wvl_point):
    # Outlier 처리
    if 0 not in wvl_point :
        # 겹치는 점 x, y축
        wvl_point = difference(wvl_point, 10, 'x', prev_wvl_point)
        
        # 아예 논 외 점 y축
        wvl_point = difference(wvl_point, 40, 'y', prev_wvl_point)
    
    return wvl_point

def difference(wvl_point, threshold, axis, prev_wvl_point):
    
    if axis == 'x':
        differences = np.diff(wvl_point[:,0])
        
        # if there is any outlier within x-axis
        result = np.any(abs(differences) <= threshold)
        if result == True:
            print("x outlier detected")
            print("\n before : ", wvl_point)
        
            # keep the index which has large difference
            indices_to_del = np.where(abs(differences) <= threshold)[0]
            wvl_point = np.delete(wvl_point, indices_to_del, axis = 0)

            print("\n after : ", wvl_point)
        return wvl_point

    else:
        differences = np.diff(wvl_point[:,1])
        
        # if there is any outlier within y-axis
        result = np.any(abs(differences) >= threshold)
        if result == True:
            print("y outlier detected")
            print("\n before : ", wvl_point)
            
            # comparison between previous y-axis points
            prev_num = np.mean(prev_wvl_point[:,1])
            diff = np.abs(wvl_point[:,1] - prev_num)
            # keep the index which has small difference between previous point
            indices_to_del = np.where(diff >= threshold)[0]
            
            wvl_point = np.delete(wvl_point, indices_to_del, axis = 0)
            print("\n after : ", wvl_point)

        return wvl_point
    
def find_order(arg, data_dir, wvl_point, wvl, n_patt, prev_wvl_point, proj_px, position):
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
    # pattern이 2000 (front) / 2050(mid) 이상부터는 grid 가 찍히지 않음
    if position == "front":
        if (wvl_point.shape[0] == 0) or (n_patt > 2000):
            wvl_point = np.zeros(shape=(arg.m_num,2))
    elif position == "mid":
        if (wvl_point.shape[0] == 0) or (n_patt > 2050):
            wvl_point = np.zeros(shape=(arg.m_num,2))
    else: # ===================================================== back spectralon 수정 =====================================================
        if (wvl_point.shape[0] == 0) or (n_patt > 2050):
            wvl_point = np.zeros(shape=(arg.m_num,2))

    # Outlier 처리
    wvl_point = out_lier(wvl_point, prev_wvl_point)
    
    # sort with x-axis to find avg of each cols
    sorted_idx = np.argsort(wvl_point[...,0], axis = 0)
    x_sort = np.take_along_axis(wvl_point[...,0], sorted_idx, axis = 0)
    y_sort = np.take_along_axis(wvl_point[...,1], sorted_idx, axis = 0)
    wvl_point[...,0], wvl_point[...,1] = x_sort, y_sort

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

    # 하나의 order만 있는경우
    if wvl_point.shape[0] == grid_pts :
        zero, first_m0, first_m2 = one_order(wvl_point, zero_arr, proj_px, position)
        
    # 두개의 m order가 있는 경우
    elif wvl_point.shape[0] == grid_pts *2:
        zero, first_m0, first_m2 = two_order(avg_t, avg_b, wvl_point, grid_pts, zero_arr)
        
    # 세개의 order가 있는 경우
    else:
        zero, first_m0, first_m2 = three_order(wvl_point, grid_pts)

    return zero, first_m0, first_m2, wvl_point
    
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    date = './calibration/dg_calibration_method2/20230728_data/front'
    point_dir = date + '_points'
    
    N_pattern = len(os.listdir(point_dir))
    # wvls = np.arange(450, 660, 50)
    # wvls = np.array([450])
    
    for i in range(N_pattern):
        detected_pts_dir = point_dir + '/pattern_%04d'%i
        # processed_points = point_process(arg, data_dir, detected_pts_dir, wvls, i, proj_px, flg)

        print('end')