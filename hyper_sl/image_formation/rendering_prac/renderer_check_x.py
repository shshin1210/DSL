import numpy as np
import torch, os, sys
from tqdm import tqdm
from scipy.io import savemat
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation.projector import Projector
from hyper_sl.image_formation.camera import Camera
from hyper_sl.image_formation import distortion
import matplotlib.pyplot as plt

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
        self.proj_focal_length = self.proj.focal_length_proj()
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
        self.xyz_proj_dg = torch.linalg.inv(self.extrinsic_diff)@self.xyz1
        self.xyz_proj_dg = self.xyz_proj_dg[:3]

        # change proj center to dg coord
        self.proj_center_dg = torch.linalg.inv(self.extrinsic_diff)@self.proj_center
        self.proj_center_dg = self.proj_center_dg[:3]
        
        # incident light, intersection points in DG coord
        # incident light dir vectors
        self.incident_dir = - self.proj_center_dg + self.xyz_proj_dg

        # make incident dir to unit vector
        self.norm = torch.norm(self.incident_dir, dim = 0)
        
        # incident light unit dir vector
        self.incident_dir_unit = self.incident_dir/self.norm
        
        self.intersection_points = self.find_intersection(self.proj_center_dg,self.incident_dir_unit)
        self.intersection_points = self.intersection_points.reshape(3,self.proj_H, self.proj_W)

        # incident light direction vectors
        self.alpha_i = self.incident_dir_unit[0]
        self.beta_i = self.incident_dir_unit[1]
        
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
        self.beta_m = torch.unsqueeze(self.beta_m, 0).repeat(self.m_n, 1)
        self.beta_m = torch.unsqueeze(self.beta_m, 1).repeat(1, self.wvls_n, 1)
        self.beta_m = self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H, self.proj_W) 
        
        self.diffracted_dir_unit = torch.stack((self.alpha_m,self.beta_m,self.z), dim = 0) 
        self.intersection_points_r = self.intersection_points.reshape(3, self.proj_H*self.proj_W) 
        self.diffracted_dir_unit_r = self.diffracted_dir_unit.reshape(3, self.m_n, self.wvls_n, self.proj_H*self.proj_W) # abz, m, wvl, H*W

        # optical center
        self.p = self.intersection_points_r.T
        self.d = self.diffracted_dir_unit_r.T
        
        # finding point p
        self.optical_center_virtual = self.get_virtual_center(self.p, self.d, self.wvls_n, self.m_n)
        # self.optical_center_virtual[0,:,0] = -self.optical_center_virtual[2,:,0]
        # self.optical_center_virtual[0,:,1] = self.optical_center_virtual[2,:,1]
        # self.optical_center_virtual[0,:,2] = self.optical_center_virtual[2,:,2]

        # optical_center_virtual shape : m_N, wvls_N, 3
        self.optical_center_virtual = torch.tensor(self.optical_center_virtual, dtype=torch.float32, device= self.device) 
        self.optical_center_proj = self.proj_center_dg.to(self.device) 
        
        ## Visualize intersection points of diffraction grating from unprojection and projection
        # sensor_z_list = torch.linspace(0.,1.6, 100)
        sensor_z_list = [0.8] # depth 0.8 Meter
        # diffraction grating coordinate에서 dg와의 intersection points
        self.point_list = torch.zeros(size=(len(sensor_z_list), self.m_n, self.wvls_n, 3, self.proj_H * self.proj_W)) # m, wvls, xyz, H*W

        ## Visualize NaN in self.z
        # for i in range(3):
        #     for j in range(25):
        #         plt.subplot()
        #         plt.imshow(self.z[i,j].detach().cpu().numpy()), plt.colorbar(), plt.title('%d order %d wvl' %(self.m_list[i],self.wvls[j]*1e9))
        #         if not os.path.exists('./vis'):
        #             os.makedirs('./vis')
        #         plt.savefig('./vis/%02d_%02d.png' %(i,j))
                
        
        # 여러 depth에 따른 visualization
        for idx, i in enumerate(sensor_z_list):
            self.sensor_z = i
            self.start = [self.intersection_points_r[0,:],self.intersection_points_r[1,:],self.intersection_points_r[2,:]]
            
            # find virtual proj sensor xyz in dg coordinate
            # 3D point XYZ 까지 닿기 위해 곱해야하는 constant t
            self.scale_sensor = (self.sensor_z-self.start[2])/self.z.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)

            # 3D point XYZ
            self.sensor_X_virtual = self.start[0] + self.scale_sensor*self.alpha_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
            self.sensor_Y_virtual = self.start[1] + self.scale_sensor*self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
            self.sensor_Z_virtual = self.start[2] + self.scale_sensor*self.z.reshape(self.m_n, self.wvls_n, self.proj_H* self.proj_W)
            self.sensor_Z_virtual = self.sensor_Z_virtual.to(self.device)

            self.XYZ = torch.stack((self.sensor_X_virtual, self.sensor_Y_virtual, self.sensor_Z_virtual), dim = 2)

            dir_vec = self.XYZ - self.optical_center_virtual.unsqueeze(dim = 3)
            norm = dir_vec.norm(dim = 2)
            
            dir_vec_unit = dir_vec/ norm.unsqueeze(dim=2)
            
            t = -self.optical_center_virtual.unsqueeze(dim = 3)[...,2,:] / dir_vec_unit[...,2,:]
            
            points = dir_vec_unit*(t.unsqueeze(dim = 2)) + self.optical_center_virtual.unsqueeze(dim = 3)
            
            self.point_list[idx] = points # m, wvl, xyz, px
        
        # save mat files
        sensor_X_real = self.intersection_points_r[0]
        sensor_Y_real = self.intersection_points_r[1]
        
        lmb = list(torch.linspace(arg.wvl_min, arg.wvl_max, arg.wvl_num))
        self.dat_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/dat"
        for m in tqdm(range(len(self.m_list))):
            for l in tqdm(range(len(lmb)), leave = False):
                
                ## Save Dg intersection points visualization
                self.dg_intersection_visualization(point_list=self.point_list, m =m,l= l, lmb=lmb)
                
                ## Check why self.z is NaN?
                # visualization of rays
                # self.vis_ray(0, -3, 10239)
                
                no_nan_idx = self.point_list[0,m,l,2].isnan() == False
                
                # Real X, Y (sensor_X_real, sensor_Y_real)
                sensor_X = sensor_X_real[no_nan_idx].data.cpu().numpy()
                sensor_Y = sensor_Y_real[no_nan_idx].data.cpu().numpy()
                
                # distorted x, y (point_list)
                dist_x = self.point_list[0, m, l, 0, no_nan_idx].data.cpu().numpy()
                dist_y = self.point_list[0, m, l, 1, no_nan_idx].data.cpu().numpy()
                
                savemat(os.path.join(self.dat_path, 'dispersion_coordinates_m%d_wvl%d.mat'%(self.m_list[m], torch.round(lmb[l]*1e9) )), {'x': dist_x, 'y': dist_y, 'xo': sensor_X, 'yo': sensor_Y})
                # savemat(os.path.join(self.dat_path, 'dispersion_coordinates_m%d_wvl%d.mat'%(self.m_list[m], torch.round(lmb[l]*1e9) )), {'x': point_list[0,m,l,0].data.cpu().numpy(), 'y': point_list[0,m,l,1].data.cpu().numpy(), 'xo': sensor_X_real.data.cpu().numpy(), 'yo': sensor_Y_real.data.cpu().numpy()})

        ### Depth Independent Intersection Points
        # visualization of x points
        self.vis_for_depth_independence(point_list=self.point_list, sensor_z_list=sensor_z_list)
        
        ## Check why self.z is NaN?
        # visualization of rays
        
    def vis_ray(self, m, l, idx):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        
        ax.set_xlim([-0.01,0.01])
        ax.set_ylim([-0.05,0.005])
        ax.set_zlim([-0.05,0.03])
        
        # optical centers
        ax.scatter(self.optical_center_virtual[:,:,0].flatten().detach().cpu().numpy(), self.optical_center_virtual[:,:,1].flatten().detach().cpu().numpy(), self.optical_center_virtual[:,:,2].flatten().detach().cpu().numpy(), color = 'pink', marker = 's')        
        # projector sensor plane dg coord
        ax.scatter(self.xyz_proj_dg[0,::500].detach().cpu().numpy(), self.xyz_proj_dg[1,::500].detach().cpu().numpy(), self.xyz_proj_dg[2,::500].detach().cpu().numpy(), color = 'blue', marker = '.', s = 0.1)
        
        # projector center dg coord
        ax.scatter(self.proj_center_dg[0].detach().cpu().numpy(),self.proj_center_dg[1].detach().cpu().numpy(),self.proj_center_dg[2].detach().cpu().numpy(), color = 'black', marker='s')
        
        # optical center dg coord (m = -1, wvl = 660nm)
        ax.scatter(self.optical_center_virtual[m,l,0].detach().cpu().numpy(), self.optical_center_virtual[m,l,1].detach().cpu().numpy(), self.optical_center_virtual[m,l,2].detach().cpu().numpy(), color = 'peru', marker = 's')
        
        # intersection point (valid ray) dg coord
        ax.scatter(self.intersection_points_r[0,idx].detach().cpu().numpy(), self.intersection_points_r[1,idx].detach().cpu().numpy(), self.intersection_points_r[2,idx].detach().cpu().numpy(), color = 'cyan', marker = '*')
        
        # 3D point
        ax.scatter(self.XYZ[m,l,0,idx].detach().cpu(), self.XYZ[m,l,1,idx].detach().cpu(), self.XYZ[m,l,2,idx].detach().cpu(), color = 'green', marker = 'o')

        # invalid intersection pts
        ax.scatter(self.point_list[0,m,l,0,idx].detach().cpu(), self.point_list[0,m,l,1,idx].detach().cpu(), self.point_list[0,m,l,2,idx].detach().cpu(), color = 'red', marker = '*')
        
        # dg coords
        ax.scatter(self.intersection_points_r[0,::100].detach().cpu(),self.intersection_points_r[1,::100].detach().cpu(), self.intersection_points_r[2,::100].detach().cpu(), color = 'purple',  marker = '.', s = 0.1)
        ax.scatter(self.point_list[0,m,l,0,::100].detach().cpu(), self.point_list[0,m,l,1,::100].detach().cpu(), self.point_list[0,m,l,2,::100].detach().cpu(), color = 'limegreen', marker = '.',s = 0.3)

        # proj center 2 intersectionpt
        X_d = [self.proj_center_dg[0].detach().cpu().numpy(), self.intersection_points_r[0,idx].detach().cpu().numpy()]
        Y_d = [self.proj_center_dg[1].detach().cpu().numpy(), self.intersection_points_r[1,idx].detach().cpu().numpy()]
        Z_d = [self.proj_center_dg[2].detach().cpu().numpy(), self.intersection_points_r[2,idx].detach().cpu().numpy()]
        ax.plot(X_d,Y_d,Z_d, color = 'black' , linewidth = 0.25)
                
        # intersectionpt 2 3Dpt
        X_d = [self.intersection_points_r[0,idx].detach().cpu().numpy(), self.XYZ[m,l,0,idx].detach().cpu()]
        Y_d = [self.intersection_points_r[1,idx].detach().cpu().numpy(), self.XYZ[m,l,1,idx].detach().cpu()]
        Z_d = [self.intersection_points_r[2,idx].detach().cpu().numpy(), self.XYZ[m,l,2,idx].detach().cpu()]
        ax.plot(X_d,Y_d,Z_d, color = 'black' , linewidth = 0.25)
           
        # 3Dpt 2 optical center pt
        X_d = [self.XYZ[m,l,0,idx].detach().cpu(), self.optical_center_virtual[m,l,0].detach().cpu().numpy()]
        Y_d = [self.XYZ[m,l,1,idx].detach().cpu(), self.optical_center_virtual[m,l,1].detach().cpu().numpy()]
        Z_d = [self.XYZ[m,l,2,idx].detach().cpu(), self.optical_center_virtual[m,l,2].detach().cpu().numpy()]
        ax.plot(X_d,Y_d,Z_d, color = 'red' , linewidth = 0.25)
        
        # 3Dpt 2 invalid pt
        X_d = [self.XYZ[m,l,0,idx].detach().cpu(), self.point_list[0,m,l,0,idx].detach().cpu()]
        Y_d = [self.XYZ[m,l,1,idx].detach().cpu(), self.point_list[0,m,l,1,idx].detach().cpu()]
        Z_d = [self.XYZ[m,l,2,idx].detach().cpu(), self.point_list[0,m,l,2,idx].detach().cpu()]
        ax.plot(X_d,Y_d,Z_d, color = 'green' , linewidth = 0.25)
        
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        
        ax.legend(['optical centers', 'proj plane', 'proj center', 'optical center', 'valid intersect point', '3D point', 'invalid intersect point', 'dg plane', 'invalid dg plane', 'proj center 2 int pts', 'valid int pts 2 3D pts', '3D pts 2 opt center', '3D pts 2 invalid pts'])

        # projector center dg coord : black
        # optical center dg coord (m = -1, wvl = 660nm) : orange
        # intersection point (valid ray) dg coord : cyan
        # invalid intersection pts : red *
        
        plt.show()  
        print()
        
    def dg_intersection_visualization(self, point_list, m, l, lmb):
        ## Save Dg intersection points visualization
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter(point_list[0,m,l,0,::100].detach().cpu().numpy(), point_list[0,m,l,1,::100].detach().cpu().numpy(), point_list[0,m,l,2,::100].detach().cpu().numpy())
        ax.set_zlim([-0.01,0.01])
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('%d order %d nm' %(self.m_list[m], torch.round(lmb[l]*1e9)))
        if m == 0:
            plt.savefig('./vis/00_minus_order_%dnm.png' %(torch.round(lmb[l]*1e9)))                
        elif m == 1:
            plt.savefig('./vis/01_zero_order_%dnm.png' %(torch.round(lmb[l]*1e9)))             
        else:
            plt.savefig('./vis/02_first_order_%dnm.png' %(torch.round(lmb[l]*1e9)))   
                           
    def vis_for_depth_independence(self, point_list, sensor_z_list):
        """
            Depth Independent Intersection Points
            visualization of x points
        
        """
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
        
        t = -proj_center_dg[2] / incident_dir_unit[2] 
        a = t.unsqueeze(dim = 0) * incident_dir_unit + proj_center_dg
        
        return a
    
    def get_alpha_m(self, m, alpha_i, lmbda):

        d = (1/500)*1e-3
        m = torch.unsqueeze(m, dim=1)
        lmbda = torch.unsqueeze(lmbda, dim = 0).to(self.device)
        alpha_i = alpha_i.unsqueeze(dim = 0).to(self.device)

        m_l_d = m*lmbda/d 
        alpha_m = - m_l_d.unsqueeze(dim = 2) + alpha_i

        return alpha_m

    def get_beta_m(self, beta_i):
        beta_m = beta_i

        return beta_m

    def get_z_m(self, alpha_m, beta_m):
        z = torch.sqrt(1 - alpha_m**2 - beta_m**2)

        return z           
        
    def get_virtual_center(self, P, dir, wvls_N, m_N):
        """P and dir are NxD arrays defining N lines.
        D is the dimension of the space. This function 
        returns the least squares intersection of the N
        """        
                
        # optical center 구할 때 
        # P : [number of px, xyz]
        # dir : [number of px, wvl, m, xyz]
        ## Optical center without NaN
        
        p_list = torch.zeros(size=(m_N, wvls_N,3))
        D_list = torch.zeros(size = (m_N, wvls_N))
        
        ### 논문에 Error 넣기!!! ====================================================================================
        
        for i in range(dir.shape[2]): # m
            for j in range(dir.shape[1]): # wvls
                dir_wvls_m = dir[:,j,i,:] # 720*720, 3
                torch_eye = torch.eye(dir_wvls_m.shape[1], device= self.device) # 3 x 3

                # exclude NaN
                dir_wvls_m_no_nan = dir_wvls_m[torch.isnan(dir_wvls_m[:,2]) == False] # number of pixels, 3
                P_no_nan = P[torch.isnan(dir_wvls_m[:,2]) == False]
                # generate the array of all projectors
                projs = torch_eye - torch.unsqueeze(dir_wvls_m_no_nan, dim = 2)*torch.unsqueeze(dir_wvls_m_no_nan, dim = 1)  # I - n*n.T                              
                        # Identity matrix 3x3        # number of pixels, 3, 3
                                        
                # generate R matrix and q vector
                R = projs.sum(axis=0) 
                q = (projs @ torch.unsqueeze(P_no_nan, dim = 2)).sum(axis=0) 

                # solve the least squares problem for the 
                # intersection point p: Rp = q
                pts = torch.linalg.lstsq(R,q,rcond=None)[0]
                
                # cross = torch.cross(pts.permute(1,0) - P_no_nan, dir_wvls_m_no_nan, dim = 1)
                # D = torch.norm(cross, p = 2, dim = 1) / torch.norm(dir_wvls_m_no_nan, p = 2, dim = 1)
                # D_list[i,j] = D.mean()
                
                pts = pts.squeeze()

                
                p_list[i,j,:] = pts

        return p_list
        
        # torch_eye = torch.eye(dir.shape[2], device=self.device)

        # projs = torch_eye - torch.unsqueeze(dir, dim = 4)*torch.unsqueeze(dir, dim = 3)
        
        # R = projs.sum(axis = 0)
        # P = P.unsqueeze(dim = 1).unsqueeze(dim = 1)
        # q = (projs @ torch.unsqueeze(P, dim = 4)).sum(axis=0) # px sum
        
        # p = torch.linalg.lstsq(R,q,rcond=None)[0]
        # p = p.squeeze().permute(1,0,2)
        
        # return p
        
if __name__ == "__main__":
   
    from hyper_sl.utils.ArgParser import Argument

    argument = Argument()
    arg = argument.parse()

    # n_scene, random, pixel_num, eval
    PixelRenderer(arg)
    
    print('end')