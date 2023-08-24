import torch
import os
from scipy.io import loadmat

class Distortion():
    def __init__(self, arg, wvls_list, depth_list):
        
        # arguments
        self.arg = arg
        self.m_list = torch.tensor([-1, 0, 1])
        self.new_m_list = torch.cat((self.m_list[:1], self.m_list[2:])).to(device = arg.device)
        self.wvls_list = wvls_list
        self.depth_list = depth_list
        
        if len(self.depth_list) > 50 :
            self.const, self.digit, self.unit = 1000, 3, "mm"
        else:
            self.const, self.digit, self.unit = 100, 2, "cm"
            
    def bring_distortion_coeff(self, dat_path):
        
        p_list = torch.zeros(size=(len(self.new_m_list), len(self.wvls_list), len(self.depth_list), 21, 2))

        # distortion coeff 합치기
        for d_i, d in enumerate(self.depth_list):
            for m_i, m in enumerate(self.new_m_list):
                for wvl_i, wvl in enumerate(self.wvls_list):
                    p = loadmat(os.path.join(dat_path, 'param_dispersion_coordinates_m%d_wvl%d_depth%d%s.mat'%(m, torch.round(self.wvls_list[wvl_i]*1e9), d* self.const, self.unit)))
                    p = p['p']
                    p = torch.tensor(p, dtype=torch.float32)

                    p_list[m_i, wvl_i, d_i] = p
        
        return p_list

    def distort_func(self, x, y, z, p, q):

        if self.unit == "cm":
            depths = torch.round(self.depth_list * 100).type(torch.int)
            z = torch.round(z * 100).type(torch.int).flatten()
        else:
            depths = torch.round(self.depth_list * 1000).type(torch.int)
            z = torch.round(z * 1000).type(torch.int).flatten()

        comparison_matrix = z[:, None] == depths

        row_indices = torch.arange(comparison_matrix.shape[1]).to(comparison_matrix.device)
        indices = torch.where(comparison_matrix, row_indices, -1)
        indices = indices.max(dim=1).values
        indices = indices.reshape(-1, self.arg.cam_H * self.arg.cam_W)
        
        x = x.unsqueeze(dim = 1).unsqueeze(dim = 1)
        y = y.unsqueeze(dim = 1).unsqueeze(dim = 1)
        
        # pick the coefficient for different depths
        p = p[:,:,indices.squeeze()]
        q = q[:,:,indices.squeeze()]

        # indices 가 -1 인 부분의 coeff를 0처리하기
        outlier_idx = (indices == -1)
        p[:,:,outlier_idx.squeeze()] = 0.
        q[:,:,outlier_idx.squeeze()] = 0.
                
        x_d = p[...,0] + p[...,1]*x + p[...,2]*y + p[...,3]*(x**2) + p[...,4]*x*y + p[...,5]*(y**2) + p[...,6]*(x**3) + p[...,7]*(x**2)*y + p[...,8]*x*(y**2) + p[...,9]*(y**3) + p[...,10]*(x**4) + p[...,11]*(x**3)*y + p[...,12]*(x**2)*(y**2) + p[...,13]*x*(y**3) + p[...,14]*(y**4) + p[...,15]*(x**5) + p[...,16]*(x**4)*y + p[...,17]*(x**3)*(y**2) + p[...,18]*(x**2)*(y**3) + p[...,19]*x*(y**4) + p[...,20]*(y**5)
        y_d = q[...,0] + q[...,1]*x + q[...,2]*y + q[...,3]*(x**2) + q[...,4]*x*y + q[...,5]*(y**2) + q[...,6]*(x**3) + q[...,7]*(x**2)*y + q[...,8]*x*(y**2) + q[...,9]*(y**3) + q[...,10]*(x**4) + q[...,11]*(x**3)*y + q[...,12]*(x**2)*(y**2) + q[...,13]*x*(y**3) + q[...,14]*(y**4) + q[...,15]*(x**5) + q[...,16]*(x**4)*y + q[...,17]*(x**3)*(y**2) + q[...,18]*(x**2)*(y**3) + q[...,19]*x*(y**4) + q[...,20]*(y**5)

        return x_d, y_d