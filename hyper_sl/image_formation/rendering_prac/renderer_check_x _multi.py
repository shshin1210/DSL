import numpy as np
import torch, os, sys
from tqdm import tqdm
from scipy.io import savemat
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation.projector_multi import Projector
from hyper_sl.image_formation.camera import Camera
from hyper_sl.image_formation import distortion

class PixelRenderer():
    """ Render for a single scene 
        which has specific # of pixels for N different patterns
        
    """
    def __init__(self, arg):
        self.arg = arg

        # device
        self.device = arg.device
        
        # arguments
        self.wvls = arg.wvls
        self.n_illum = arg.illum_num
        
        # classes
        self.proj = Projector(arg, device= self.device)
        self.cam = Camera(arg)
        self.load_data = load_data.load_data(arg)
        self.dist = distortion
        self.normalize = normalize.normalization
        
        self.load_data = load_data.load_data(arg)
        self.noise = noise.GaussianNoise(0, arg.noise_std, arg.device)
        self.normalize = normalize
        
        # cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.cam_sensor_width = arg.sensor_width*1e-3
        self.cam_focal_length = arg.focal_length*1e-3
        self.crf_cam_path = arg.camera_response
        self.CRF_cam = torch.tensor(np.load(os.path.join(self.crf_cam_path, 'CRF_cam.npy'))).to(self.device)
        self.cam_pitch = arg.cam_pitch

        # proj
        self.proj_sensor_diag = arg.sensor_diag_proj *1e-3
        self.proj_focal_length = arg.focal_length_proj *1e-3
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.crf_proj_path = arg.projector_response
        self.CRF_proj = torch.tensor(np.load(os.path.join(self.crf_proj_path, 'CRF_proj.npy'))).to(self.device)
        self.sensor_height_proj = (torch.sin(torch.atan2(torch.tensor(self.proj_H),torch.tensor(self.proj_W)))*self.proj_sensor_diag)
        self.proj_pitch = (self.sensor_height_proj/ (self.proj_H))
        self.intrinsic_proj_real = self.proj.intrinsic_proj_real()
        
        # path
        self.dat_dir = arg.dat_dir
        
        # m order, wvls
        self.m_list = arg.m_list.to(device=self.device)
        self.m_n = len(self.m_list)
        self.wvls = arg.wvls.to(device=self.device)
        self.wvls_n = len(self.wvls)
        
        # extrinsic matrix
        self.extrinsic_proj_real = self.proj.extrinsic_proj_real()
        self.extrinsic_diff = self.proj.extrinsic_diff()
        
        # projector sensor plane
        self.xyz1, self.proj_center = self.proj_sensor_plane()
        
        # change proj sensor to dg coord
        self.xyz_proj_dg = torch.linalg.inv(self.extrinsic_diff)@self.xyz1 # m, wvl, xyz1, px
        self.xyz_proj_dg = self.xyz_proj_dg[:,:,:3]

        # change proj center to dg coord
        self.proj_center_dg = torch.linalg.inv(self.extrinsic_diff)@self.proj_center # m, wvl, xyz1, px
        self.proj_center_dg = self.proj_center_dg[:,:,:3]
        
        # incident light, intersection points in DG coord
        # incident light dir vectors
        self.incident_dir = - self.proj_center_dg + self.xyz_proj_dg

        # make incident dir to unit vector
        self.norm = torch.norm(self.incident_dir, dim = 2) # m, wvl, xyz1, px
        
        # incident light unit dir vector
        self.incident_dir_unit = self.incident_dir/self.norm.unsqueeze(dim = 2)
        
        self.intersection_points = self.find_intersection(self.proj_center_dg,self.incident_dir_unit)
        self.intersection_points = self.intersection_points.reshape(self.m_n, self.wvls_n, 3, self.proj_H, self.proj_W) # m, wvl, xyz1, H, W
        self.intersection_points = self.intersection_points.permute(2,0,1,3,4) # xyz1, wvl, m, H, W

        # incident light direction vectors
        self.alpha_i = self.incident_dir_unit[:,:,0]
        self.beta_i = self.incident_dir_unit[:,:,1]
        
        self.CRF_cam = torch.tensor(self.cam.get_CRF()).to(self.device)
        self.CRF_proj = torch.tensor(self.proj.get_PRF()).to(self.device)

        #### Scene independent variables
        # compute direction cosines
        self.alpha_m = self.get_alpha_m(alpha_i=self.alpha_i, m= self.m_list, lmbda= self.wvls)
        self.beta_m = self.get_beta_m(beta_i=self.beta_i) 
        self.z = self.get_z_m(self.alpha_m, self.beta_m)
        
        self.alpha_m = self.alpha_m.reshape(self.m_n, self.wvls_n, self.proj_H, self.proj_W) 
        self.z = self.z.reshape(self.m_n, self.wvls_n, self.proj_H, self.proj_W)

        # for m = [-1,0,1]
        self.beta_m = self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H, self.proj_W) 
        
        self.diffracted_dir_unit = torch.stack((self.alpha_m,self.beta_m,self.z), dim = 0) 
        self.intersection_points_r = self.intersection_points.reshape(3, self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
        self.diffracted_dir_unit_r = self.diffracted_dir_unit.reshape(3, self.m_n, self.wvls_n, self.proj_H*self.proj_W) # abz, m, wvl, H*W

        # optical center
        self.p = self.intersection_points_r.T # xyz, m, wvl, px
        self.d = self.diffracted_dir_unit_r.T
        
        # finding point p
        self.optical_center_virtual = self.proj.get_virtual_center(self.p, self.d, self.wvls_n, self.m_n)
        
        # optical_center_virtual shape : m_N, wvls_N, 3
        self.optical_center_virtual = torch.tensor(self.optical_center_virtual, dtype=torch.float32, device= self.device) 
        self.optical_center_proj = self.proj_center_dg.to(self.device) 
        
        # virtual projector pixels
        # sensor_z_list = torch.linspace(0.,1.6,100)
        sensor_z_list = [0.8]
        # diffraction grating coordinate에서 dg와의 intersection points
        point_list = torch.zeros(size=(len(sensor_z_list), self.m_n, self.wvls_n, 3, self.proj_H * self.proj_W)) # m, wvls, xyz, H*W

        for idx, i in enumerate(sensor_z_list):
            self.sensor_z = i
            self.start = [self.intersection_points_r[0,:],self.intersection_points_r[1,:],self.intersection_points_r[2,:]]
            
            # find virtual proj sensor xyz in dg coordinate
            self.scale_sensor = (self.sensor_z-self.start[2])/self.z.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)

            self.sensor_X_virtual = self.start[0] + self.scale_sensor*self.alpha_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
            self.sensor_Y_virtual = self.start[1] + self.scale_sensor*self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
            self.sensor_Z_virtual = self.start[2] + self.scale_sensor*self.z.reshape(self.m_n, self.wvls_n, self.proj_H* self.proj_W)
            self.sensor_Z_virtual = self.sensor_Z_virtual.to(self.device)

            XYZ = torch.stack((self.sensor_X_virtual, self.sensor_Y_virtual, self.sensor_Z_virtual), dim = 2)

            dir_vec = XYZ - self.optical_center_virtual.unsqueeze(dim = 3)
            norm = dir_vec.norm(dim = 2)
            
            dir_vec_unit = dir_vec/ norm.unsqueeze(dim=2)
            
            t = -self.optical_center_virtual.unsqueeze(dim =3)[...,2,:] / dir_vec_unit[...,2,:]
            
            points = dir_vec_unit*(t.unsqueeze(dim = 2)) + self.optical_center_virtual.unsqueeze(dim = 3)
            
            point_list[idx] = points
        
        # save mat files
        sensor_X_real = self.intersection_points_r[0]
        sensor_Y_real = self.intersection_points_r[1]
        
        lmb = list(torch.linspace(arg.wvl_min, arg.wvl_max, arg.wvl_num))
        self.dat_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/dat"
        for m in tqdm(range(len(self.m_list))):
            for i in tqdm(range(len(lmb)), leave = False):
                
                savemat(os.path.join(self.dat_path, 'dispersion_coordinates_m%d_wvl%d.mat'%(self.m_list[m], torch.round(lmb[i]*1e9))), {'x': point_list[0,m,i,0].data.cpu().numpy(), 'y': point_list[0,m,i,1].data.cpu().numpy(), 'xo': sensor_X_real[m,i].data.cpu().numpy(), 'yo': sensor_Y_real[m,i].data.cpu().numpy()})
                

        # visualization of x points
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x_points = torch.zeros(size=(len(point_list),))
        y_points = torch.zeros(size=(len(point_list),))
        
        
        for i in range(len(point_list)):
            y_points[i] = point_list[i,2,-4,0,191222]
            if i % 10 == 0:
                print(y_points[i].detach().cpu().numpy())
        
        print(self.intersection_points_r[0,191222].detach().cpu().numpy())
        
        plt.plot(sensor_z_list, y_points, '-')
        # plt.plot(self.intersection_points_r[0, 191222].detach().cpu(), '.', color = 'red')
        plt.show()
        
    def proj_sensor_plane(self):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """
        #  proj sensor
        xs = torch.linspace(0,self.proj_H-1, self.proj_H)
        ys = torch.linspace(0,self.proj_W-1, self.proj_W)
        r, c = torch.meshgrid(xs, ys, indexing='ij')
        
        c, r = c.flatten(), r.flatten()
        ones = torch.ones_like(c)
        cr1 = torch.stack((c,r,ones), dim = 0)
        xyz = (torch.linalg.inv(self.intrinsic_proj_real)@(cr1*self.proj_focal_length))

        # proj_center
        proj_center = torch.zeros(size=(4,1), device=self.device)
        proj_center[3,0] = 1

        # make projector sensor xyz1 vector
        xyz1 = torch.concat((xyz, torch.ones(size=(1, xyz.shape[1]))), dim = 0)
        xyz1 = xyz1.to(self.device)
        
        return xyz1, proj_center
    
    def find_intersection(self, proj_center_dg, incident_dir_unit):
        """ find the intersection of incident rays and diffraction gratings
        
            returns : intersection points
        """
        
        # incident_dir_unit m, wvl, xyz1, px
        t = -proj_center_dg[:,:,2] / incident_dir_unit[:,:,2] 
        a = t.unsqueeze(dim = 2) * incident_dir_unit + proj_center_dg
        
        return a

    def get_alpha_m(self, m, alpha_i, lmbda):

        d = (1/500)*1e-3
        m = torch.unsqueeze(m, dim=1)
        lmbda = torch.unsqueeze(lmbda, dim = 0).to(self.device)
        alpha_i = alpha_i.to(self.device)

        m_l_d = m*lmbda/d 
        alpha_m = - m_l_d.unsqueeze(dim = 2) + alpha_i

        return alpha_m

    def get_beta_m(self, beta_i):
        beta_m = beta_i

        return beta_m

    def get_z_m(self, alpha_m, beta_m):
        z = torch.sqrt(1 - alpha_m**2 - beta_m**2)

        return z
        
        
if __name__ == "__main__":
   
    from hyper_sl.utils.ArgParser import Argument

    argument = Argument()
    arg = argument.parse()

    # n_scene, random, pixel_num, eval
    PixelRenderer(arg)
    
    print('end')