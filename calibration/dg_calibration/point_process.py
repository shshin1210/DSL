import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
import numpy as np
from scipy.io import loadmat

def point_process(arg, total_dir, pattern_dir, wvls, n_patt):
    # processing points
    # if reorder == True:
    #     processed_points = reorder_points(arg, pattern_dir, wvls)
    
    pixel_num = loadmat(os.path.join(pattern_dir,'%dnm_undistort_centroid.mat' %(wvls[0])))['s'].shape[0] // 2
    processed_points = np.zeros(shape=(arg.m_num, len(wvls), pixel_num, 2))
    
    for w in range(len(wvls)):
        wvl_point = np.array(loadmat(os.path.join(pattern_dir,'%dnm_undistort_centroid.mat' %(wvls[w])))['s'])            
        wvl_point = np.array([pts[0][0][0] for pts in wvl_point])
        
        zero, first = find_order(total_dir, wvl_point, wvls[w], n_patt)
        
        # zero order
        processed_points[1] = zero[np.newaxis,:,:]
        
        if n_patt == 0:
            # first order
            processed_points[0, w, :, :] = np.concatenate((first[:3], first[:3]), axis = 0)
            processed_points[2, w, :, :] = np.concatenate((first[3:], first[3:]), axis = 0)
        else:
            processed_points[0, w, :, :] = first
            processed_points[2, w, :, :] = first
            
    return processed_points
    
def find_order(total_dir, wvl_point, wvl, n_patt):
    dir = total_dir + 'test_2023_05_15_17_34_processed/'
    img = cv2.imread(dir+ 'pattern_%02d/%03dnm_undistort.png'%(n_patt, wvl), 0)
    
    pts = np.array([img[i[1].astype(np.int16) ,i[0].astype(np.int16)] for i in wvl_point])
    avg = np.average(pts) - 5

    first = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value < avg])
    zero = np.array([wvl_point[idx] for idx, value in enumerate(pts) if value > avg])
    
    return zero, first


if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    total_dir = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/"
    point_dir = total_dir + 'test_2023_05_15_17_34_points'
    
    N_pattern = 3
    wvls = np.arange(450, 660, 50)
    
    for i in range(N_pattern):
        pattern_dir = point_dir + '/pattern_%02d'%i
        processed_points = point_process(arg, total_dir, pattern_dir, wvls, i)
        
        
        print('end')