import torch
import os
from scipy.io import loadmat

class Distortion():
    def __init__(self, arg):
        
        # arguments
        self.arg = arg
        self.m_list = torch.tensor([-1, 1])
        self.wvls_list = self.wvls
        self.depth_list = arg.depth_list
        
        # dir
        self.dat_path = "../../dataset/image_formation/dat/method2/interpolated"
        
    def bring_distortion_coeff(self):
        
        p_list = torch.zeros(size=(len(self.m_list), len(self.wvls_list), len(self.depth_list), 21, 2))

        # distortion coeff 합치기
        for d_i, d in enumerate(self.depth_list):
            for m_i, m in enumerate(self.m_list):
                for wvl_i, wvl in enumerate(self.wvls_list):
                    p = loadmat(os.path.join(self.dat_path, 'param_dispersion_coordinates_m%d_wvl%d_depth%dcm.mat'%(m, wvl, d*100)))
                    p = p['p']
                    p = torch.tensor(p, dtype=torch.float32)

                    p_list[m_i, wvl_i] = p
        
        return p_list

    def distort_func(self, x, y, p, q, N = 5):

        x = x.unsqueeze(dim = 1).unsqueeze(dim = 1)
        y = y.unsqueeze(dim = 1).unsqueeze(dim = 1)
        
        p = p.unsqueeze(dim = 3)
        q = q.unsqueeze(dim = 3)
        
        print(x.shape, y.shape, p.shape, q.shape)
        
        x_d = p[...,0] + p[...,1]*x + p[...,2]*y + p[...,3]*(x**2) + p[...,4]*x*y + p[...,5]*(y**2) + p[...,6]*(x**3) + p[...,7]*(x**2)*y + p[...,8]*x*(y**2) + p[...,9]*(y**3) + p[...,10]*(x**4) + p[...,11]*(x**3)*y + p[...,12]*(x**2)*(y**2) + p[...,13]*x*(y**3) + p[...,14]*(y**4) + p[...,15]*(x**5) + p[...,16]*(x**4)*y + p[...,17]*(x**3)*(y**2) + p[...,18]*(x**2)*(y**3) + p[...,19]*x*(y**4) + p[...,20]*(y**5)
        y_d = q[...,0] + q[...,1]*x + q[...,2]*y + q[...,3]*(x**2) + q[...,4]*x*y + q[...,5]*(y**2) + q[...,6]*(x**3) + q[...,7]*(x**2)*y + q[...,8]*x*(y**2) + q[...,9]*(y**3) + q[...,10]*(x**4) + q[...,11]*(x**3)*y + q[...,12]*(x**2)*(y**2) + q[...,13]*x*(y**3) + q[...,14]*(y**4) + q[...,15]*(x**5) + q[...,16]*(x**4)*y + q[...,17]*(x**3)*(y**2) + q[...,18]*(x**2)*(y**3) + q[...,19]*x*(y**4) + q[...,20]*(y**5)

        return x_d, y_d