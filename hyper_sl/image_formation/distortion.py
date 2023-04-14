import torch
import os
from scipy.io import loadmat

def bring_distortion_coeff(arg, m_list, wvls, dat_path):
    
    p_list = torch.zeros(size=(len(arg.m_list), len(arg.wvls), 2, 21))

    # distortion coeff 합치기
    for m_i, m in enumerate(m_list):
        for wvl_i, wvl in enumerate(wvls):
            wvl_item = torch.tensor(wvl.item())
            wvl_item = torch.round(wvl_item*1e9)
            p = loadmat(os.path.join(dat_path, 'param_dispersion_coordinates_m%d_wvl%d.mat'%(m, wvl_item)))
            p = p['p']
            p = torch.tensor(p, dtype=torch.float32)

            p_list[m_i, wvl_i] = p
    
    return p_list

def distort_func(x, y, p, q, N = 5): # p = p[:,0], q = p[:,1]
    cnt=0
    
    p = torch.unsqueeze(p, dim = 3)
    q = torch.unsqueeze(q, dim = 3)
    x_d, y_d = torch.zeros_like(x), torch.zeros_like(y) 

    x_d = p[:,:,0] + p[:,:,1]*x + p[:,:,2]*y + p[:,:,3]*(x**2) + p[:,:,4]*x*y + p[:,:,5]*(y**2) + p[:,:,6]*(x**3) + p[:,:,7]*(x**2)*y + p[:,:,8]*x*(y**2) + p[:,:,9]*(y**3) + p[:,:,10]*(x**4) + p[:,:,11]*(x**3)*y + p[:,:,12]*(x**2)*(y**2) + p[:,:,13]*x*(y**3) + p[:,:,14]*(y**4) + p[:,:,15]*(x**5) + p[:,:,16]*(x**4)*y + p[:,:,17]*(x**3)*(y**2) + p[:,:,18]*(x**2)*(y**3) + p[:,:,19]*x*(y**4) + p[:,:,20]*(y**5)
    y_d = q[:,:,0] + q[:,:,1]*x + q[:,:,2]*y + q[:,:,3]*(x**2) + q[:,:,4]*x*y + q[:,:,5]*(y**2) + q[:,:,6]*(x**3) + q[:,:,7]*(x**2)*y + q[:,:,8]*x*(y**2) + q[:,:,9]*(y**3) + q[:,:,10]*(x**4) + q[:,:,11]*(x**3)*y + q[:,:,12]*(x**2)*(y**2) + q[:,:,13]*x*(y**3) + q[:,:,14]*(y**4) + q[:,:,15]*(x**5) + q[:,:,16]*(x**4)*y + q[:,:,17]*(x**3)*(y**2) + q[:,:,18]*(x**2)*(y**3) + q[:,:,19]*x*(y**4) + q[:,:,20]*(y**5)

    return x_d, y_d