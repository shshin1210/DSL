import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
from scipy.io import loadmat

def point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, n_patt):

    pixel_num = grid_pts
    processed_points = np.zeros(shape=(arg.m_num, len(wvls), pixel_num, 2))
    
    for w in range(len(wvls)):
        wvl_point = np.array(loadmat(os.path.join(pattern_dir,'%dnm_centroid.mat' %(wvls[w])))['s'])            
        wvl_point = np.array([pts[0][0][0] for pts in wvl_point])
        
        zero, first, bool = find_order(grid_pts, total_dir, date, wvl_point, wvls[w], n_patt)
        
        # zero order
        processed_points[1] = zero[np.newaxis,:,:]
        
        if bool == False:
            processed_points[0, w, :, :] = first
        else:
            processed_points[2, w, :, :] = first
    
    # sorting
    sorted_idx = np.argsort(-processed_points[...,1], axis = 2)
    x_sort = np.take_along_axis(processed_points[...,0], sorted_idx, axis = 2)
    y_sort = np.take_along_axis(processed_points[...,1], sorted_idx, axis = 2)
    
    processed_points[...,0], processed_points[...,1] = x_sort, y_sort
    
    return processed_points
    
def find_order(grid_pts, total_dir, date, wvl_point, wvl, n_patt):
    dir = total_dir + date + '_processed/'
    img = cv2.imread(dir+ 'pattern_%02d/%03dnm.png'%(n_patt, wvl))
    img_m = img.mean(axis = 2)
        
    pts = np.array([img_m[i[1].astype(np.int16) ,i[0].astype(np.int16)] for i in wvl_point])

    # proj emission ftn 때문에 500nm intensity low
    # if (n_patt == 0) and (wvl == 500) and (len(pts) > grid_pts*2 -1):
    if (len(pts) > grid_pts*2 -1) and (n_patt < 5):
        pts[4] = 160
        
    # zero order / first order
    if len(pts) < grid_pts + 1:
        zero = np.array([wvl_point[idx] for idx, _ in enumerate(pts)])
        first = np.zeros_like(zero)
        
    # 안찍힌 점 예외처리
    if len(pts) < grid_pts *2 :
        avg = np.average(pts) - 8
        zero = np.zeros(shape = (grid_pts, 2))
        # 1st order 안찍힘
        if len(pts) == grid_pts *2 -1:
            zero = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value >= avg ])
            first = np.zeros_like(zero)
            first[:-1] = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value < avg ])
        # 1st order & 0th order 안찍힘
        else:
            zero[:-1] = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value >= avg ])
            first = np.zeros_like(zero)
            first[:-1] = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value < avg ])
    else:
        if n_patt == 9:
            pts[5] = 130
        avg = np.average(pts) - 8
        first = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value < avg ])
        zero = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value >= avg ])
    
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
    date = 'test_2023_06_24_13_40'
    point_dir = total_dir + date + '_points'
    
    N_pattern = len(os.listdir(point_dir))
    wvls = np.arange(450, 660, 50)
    grid_pts = 5
    
    for i in range(N_pattern):
        pattern_dir = point_dir + '/pattern_%02d'%i
        processed_points = point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, i)

        print('end')