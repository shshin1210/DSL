import torch
import sys
# sys.path.append('C:/Users/mainuser/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import load_data

from hyper_sl.image_formation.projector import Projector
from hyper_sl.image_formation.camera import Camera

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = 'cuda'

# rotation matrix of diffraction gratings

class PixelRenderer():
    """ Render for a single scene 
        which has specific # of pixels for N different patterns
        
    """
    def __init__(self, arg, opt_param, depth):
        self.arg = arg
    
        
        # device
        self.device = device
        
        # arguments
        self.wvls = arg.wvls
        # illum 개수 1개
        self.n_illum = 1
        self.opt_param = opt_param
        
        # classes
        self.proj = Projector(arg, device= self.device)
        self.cam = Camera(arg)
        
        self.load_data = load_data.load_data(arg)
        
        # cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.cam_focal_length = arg.focal_length*1e-3
        self.int_cam = self.cam.intrinsic_cam()
        
        # proj
        self.proj_focal_length = arg.focal_length_proj *1e-3
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.proj_pitch = 7.9559e-06
        
        # path
        self.dat_dir = arg.dat_dir
        
        # m order, wvls
        self.m_list = arg.m_list.to(device=self.device)
        self.m_n = len(self.m_list)
        self.wvls = torch.tensor([450,500,550,600,650])*1e-9
        self.wvls = self.wvls.to(device=device)
        self.wvls_n = len(self.wvls)
        
        # extrinsic matrix
        self.extrinsic_proj_real = self.proj.extrinsic_proj_real() 
        
        # make sph2cart
        self.u = self.sph2cart(opt_param[0], opt_param[1])
        # self.u.retain_grad()

        self.extrinsic_diff = self.ext_diff(self.u, opt_param[3], opt_param[2])   #proj2dg
        # self.extrinsic_diff.retain_grad()

        # projector sensor plane
        self.xyz1, self.proj_center = self.proj_sensor_plane()
        self.pixel_num = self.xyz1.shape[1]
        
        # change proj sensor plane to dg coord
        # self.xyz_proj_dg = torch.linalg.inv(self.extrinsic_diff)@self.xyz1
        self.xyz_proj_dg = self.extrinsic_diff@self.xyz1
        self.xyz_proj_dg = self.xyz_proj_dg[:3]

        # change proj center to dg coord
        # self.proj_center_dg = torch.linalg.inv(self.extrinsic_diff)@self.proj_center
        self.proj_center_dg = (self.extrinsic_diff)@self.proj_center
        self.proj_center_dg = self.proj_center_dg[:3]
        
        # incident light, intersection points in DG coord
        # incident light dir vectors
        self.incident_dir = self.xyz_proj_dg -self.proj_center_dg
        # incident light dir in proj coord
        self.incident_dir_proj = self.xyz1[:3] - self.proj_center[:3]
        
        # make incident dir to unit vector
        self.norm = torch.norm(self.incident_dir, dim = 0)
        # make incident dir to unit vector in proj coord
        self.norm_proj = torch.norm(self.incident_dir_proj, dim = 0)
        
        # incident light unit dir vector
        self.incident_dir_unit = self.incident_dir/self.norm
        # proj coord
        self.incident_dir_unit_proj = self.incident_dir/self.norm_proj
        
        # intersection points in dg coord
        self.intersection_points = self.find_intersection(self.proj_center_dg,self.incident_dir_unit)
        
        # intersection points in dg coord to proj coord
        self.ones = torch.ones(size=(self.intersection_points.shape[1],), device = self.device).unsqueeze(dim = 0)
        self.intersection_points1 = torch.concat((self.intersection_points, self.ones), dim = 0)
        # self.intersection_points_proj = self.extrinsic_diff@self.intersection_points1
        self.intersection_points_proj = torch.linalg.inv(self.extrinsic_diff)@self.intersection_points1
        
        # incident light direction vectors in projector coord
        self.alpha_i = self.incident_dir_unit_proj[0,:]
        self.beta_i = self.incident_dir_unit_proj[1,:]

        # compute direction cosines in proj coord
        self.alpha_m = self.get_alpha_m(alpha_i=self.alpha_i, m= self.m_list, lmbda= self.wvls)
        self.beta_m = self.get_beta_m(beta_i=self.beta_i)
        self.z = self.get_z_m(self.alpha_m, self.beta_m)
        
        self.alpha_m = self.alpha_m.reshape(self.m_n, self.wvls_n, self.pixel_num) 
        self.z = self.z.reshape(self.m_n, self.wvls_n, self.pixel_num).to(self.device)

        # for m = [-1,0,1]
        self.beta_m = torch.unsqueeze(self.beta_m, 0).repeat(self.m_n, 1)
        self.beta_m = torch.unsqueeze(self.beta_m, 1).repeat(1, self.wvls_n, 1)
        self.beta_m = self.beta_m.reshape(self.m_n, self.wvls_n, self.pixel_num) 

    def cart2sph(self,x, y, z):
        azimuth = np.arctan2(y,x)
        elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
        #r = np.sqrt(x**2 + y**2 + z**2)
        return elevation, azimuth

    def sph2cart(self, elevation, azimuth):
        """
            change spherical coord to cart coord
            input : elevation, azimuth
            output : normal vector(not unit)
        
        """
        r = 1
        x = r * torch.cos(elevation) * torch.cos(azimuth)
        y = r * torch.cos(elevation) * torch.sin(azimuth)
        z = r * torch.sin(elevation)
        
        normal_vec = torch.stack((x,y,z), dim = 0)
        
        return normal_vec
    def rotation_matrix(self,u, k):
        """
            input : vectors
            calculate the rotation matrix between two vectors
            output : rotation matrix
        
        """

        cos_k = torch.cos(k)
        sin_k = torch.sin(k)
        self.u_x, u_y, u_z = u[0], u[1], u[2]
        # self.u_x.retain_grad()
        
        rot_mat = torch.zeros((3,3), device=self.device)
        rot_mat[0,0] = cos_k + (self.u_x**2)*(1-cos_k)
        rot_mat[0,1] = self.u_x*u_y*(1-cos_k) - u_z*sin_k
        rot_mat[0,2] = self.u_x*u_z*(1-cos_k) + u_y*sin_k
        rot_mat[1,0] = u_y*self.u_x*(1-cos_k) + u_z*sin_k
        rot_mat[1,1] = cos_k + (u_y**2)*(1-cos_k)
        rot_mat[1,2] =  u_y*u_z*(1-cos_k)- self.u_x*sin_k
        rot_mat[2,0] = u_z*self.u_x*(1-cos_k) - u_y*sin_k
        rot_mat[2,1] = u_z*u_y*(1-cos_k) + self.u_x*sin_k
        rot_mat[2,2] = cos_k + (u_z**2)*(1-cos_k)

        return rot_mat

    def ext_diff(self, u, k, point):
        """
            diffraction grating extrinsic matrix (dg coord = inv(ext diff) @ proj plane coord)
            
            rotation matrix : normal vector of diffraction gratings & normal vector of projector plane
        
        """
        extrinsic_diff = torch.zeros((4,4), device= device)
        
        self.rot_mat = self.rotation_matrix(u, k)
        # self.rot_mat.retain_grad()

        # rotation
        extrinsic_diff[:3,:3] = self.rot_mat

        # translate
        extrinsic_diff[0,3] = 0
        extrinsic_diff[1,3] = 0
        extrinsic_diff[2,3] = -point
        extrinsic_diff[3,3] = 1
                
        return extrinsic_diff


    def render(self, depth, i):        
        # depth for m = [-1,0,1], wvl
        depth = (depth).repeat(self.m_n, 1)
        depth = torch.unsqueeze(depth, 1).repeat(1, self.wvls_n, 1)
        depth = depth.reshape(self.m_n, self.wvls_n, self.pixel_num) 
        
        # constant where z equals to depth value
        t = (depth-self.intersection_points_proj[2])/self.z
        
        # 3D XYZ points
        self.X, self.Y, self.Z = self.intersection_points_proj[0] + self.alpha_m*t, self.intersection_points_proj[1] + self.beta_m*t, self.intersection_points_proj[2] + self.z*t

        ##### CAM COORDINATE
        # project XYZ proj coord onto cam plane / uv : cam H, W / xy : real cam coord coord
        self.xy_cam, uv_cam = self.projection(self.X,self.Y,self.Z)
        
        # predicted xy cam coords
        self.xy_cam = self.xy_cam.reshape(3,self.m_n, self.wvls_n, self.pixel_num)
        # predicted xy cam coords for m = 1, m = -1
        xyz_cam_m_1, xyz_cam_m1 = self.xy_cam[:,0], self.xy_cam[:,2]
        self.xyz_cam_m_1, self.xyz_cam_m1 = xyz_cam_m_1[...,:3], xyz_cam_m1[...,3:]
        
        # predicted uv cam coords
        uv_cam = uv_cam.reshape(3,self.m_n, self.wvls_n, self.pixel_num)[:2]
        uv_cam_m_1, uv_cam_m1 = uv_cam[:,0], uv_cam[:,2]
        uv_cam_m_1, uv_cam_m1 = uv_cam_m_1[...,:3], uv_cam_m1[...,3:]
        
        # ground truth m = 1, m = -1 
        self.xy_wvl_m1_real = torch.tensor([[[380,382,384],  # 1, 2, 3
                                        [335,338,341],      # 5개 : wvl
                                        [293,296,300],      # 2개 : x & y
                                        [251,254,256],
                                        [208,212,214]],
                                       
                                       [[150,315,483],
                                        [151,315,484],
                                        [149,314,483],
                                        [149,315,483],
                                        [148,316,484]]
                                       ], device= device)

        self.xy_wvl_m_1_real = torch.tensor([[[474,470,468],   # 4, 5, 6
                                        [515,513,509],
                                        [557,553,551],
                                        [599,594,593],
                                        [640,634,635]],
                                        
                                       [[149,314,484],
                                        [148,313,484],
                                        [148,313,483],
                                        [148,314,482],
                                        [148,314,484]]
                                       ], device= device)

        # ground truth real xy cam coord
        xyz_real = self.real_to_xy(self.xy_wvl_m_1_real, self.xy_wvl_m1_real)
        self.xyz_real = xyz_real.reshape(3, 2, 5, 3)
            
        if i % 100 == 0:
            fig, ax = plt.subplots(figsize = (10,5))
            plt.plot(self.xy_wvl_m_1_real[0].detach().cpu().numpy().flatten(), self.xy_wvl_m_1_real[1].detach().cpu().numpy().flatten(), '.', label = 'gt of m = -1')
            plt.plot(self.xy_wvl_m1_real[0].detach().cpu().numpy().flatten(), self.xy_wvl_m1_real[1].detach().cpu().numpy().flatten(), '.', label = 'gt of m = 1')
            plt.plot(uv_cam_m_1[0].detach().cpu().numpy().flatten(), uv_cam_m_1[1].detach().cpu().numpy().flatten(), '.', label = 'pred m = -1')
            plt.plot(uv_cam_m1[0].detach().cpu().numpy().flatten(), uv_cam_m1[1].detach().cpu().numpy().flatten(), '.', label = 'pred m = 1')
            plt.axis('equal')
            rect = patches.Rectangle((0,0), 890,580, linewidth = 1, edgecolor = 'r', facecolor='none')
            ax.add_patch(rect)
            ax.legend(loc = 'best')
            plt.savefig('./dg_cal/real_and_uv.png')
        
        # self.visualization()
        
        return uv_cam_m1, uv_cam_m_1, self.xy_wvl_m1_real, self.xy_wvl_m_1_real
    
    # gt coord camera planes
    def real_to_xy(self, xy_m_1, xy_m1): 
        """
            ground truth coordinate uv coord to real cam plane xy coord
        """
        xy_m1andm_1 = torch.stack((xy_m_1, xy_m1), dim = 1) # 2 2 5 3
        ones = torch.ones(size=(1,2,5,3), device=device)
        
        xy1_m1andm_1 = torch.concat((xy_m1andm_1, ones), dim = 0).reshape(3, -1) # 3 2 5 3
        
        xyz_c = (torch.linalg.inv(self.intrinsic_cam()).to(self.device)@xy1_m1andm_1.float())
        xyz_c[2] = self.cam_focal_length
        
        return xyz_c       
    
    def projection(self, X,Y,Z):
        """
            proj coord to world/cam coord
        """

        # focal_length_proj_virtual : 3, 77

        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 0)

        # XYZ 3D points proj coord -> cam coord                   
        XYZ_cam = (self.extrinsic_proj_real)@XYZ1

        # uv cam coord
        uv_cam = (self.int_cam.to(device))@XYZ_cam[:3]
        uv_cam = uv_cam / uv_cam[2]
        
        # uv to xy coord
        uv_cam = uv_cam.to(self.device)
        
        xyz_c = (torch.linalg.inv(self.intrinsic_cam()).to(self.device)@uv_cam)
        xyz_c[2] = self.cam_focal_length

        return xyz_c , uv_cam
    
    def intrinsic_cam(self):
        intrinsic_cam = torch.tensor([[1.7471120984549243e+03/ self.cam_focal_length, 0.00000000e+00 , 4.3552404635908243e+02],
                                    [0.00000000e+00 ,1.7562111249245049e+03/ self.cam_focal_length , 3.3663669106446793e+02],
                                    [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
            
        return intrinsic_cam
    
    def proj_sensor_plane(self):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """

        uv1_p = torch.tensor([[152.,101.,1.],[151.,201.,1.],[150.,301.,1,],[559.,95.,1,],[556.,199.,1,],[555.,300.,1.]]).T.to(self.device)
        
        # to real projector plane size
        xyz_p = (torch.linalg.inv(self.intrinsic_proj_real()).to(self.device)@uv1_p)

        xyz_p[2] = self.proj_focal_length  # TODO: why?

        # proj_center
        proj_center = torch.zeros(size=(4,1), device=self.device)
        proj_center[3,0] = 1

        # make projector sensor xyz1 vector
        xyz1 = torch.concat((xyz_p, torch.ones(size=(xyz_p.shape[1],), device = self.device).unsqueeze(dim =0)), dim = 0)
        xyz1 = xyz1.to(self.device)
        
        return xyz1, proj_center
    
    def intrinsic_proj_real(self):
        """
            example:
            
            ones = torch.ones_like(r, device=device)
            xyz_p = torch.stack((r*depth,c*depth,ones*depth), dim = 0)
            XYZ = torch.linalg.inv(proj_int)@xyz_p
            
        """
        # TODO: divided by focal length?
        intrinsic_proj_real = torch.tensor([[1.0205325617292132e+03/ self.proj_focal_length, 0.00000000e+00, 2.7398003835418473e+02],
                                            [0.00000000e+00,1.0204965778160497e+03/ self.proj_focal_length, 3.2068450274841155e+02],
                                            [0.00000000e+00,0.00000000e+00, 1.00000000e+00]])
        
        
        return intrinsic_proj_real
    
    def find_intersection(self, proj_center_dg, incident_dir_unit):
        """ find the intersection of incident rays and diffraction gratings
        
            returns : intersection points
        """
        
        t = (- proj_center_dg[2]) / incident_dir_unit[2]
        a = proj_center_dg + t * incident_dir_unit
        
        return a

    def get_alpha_m(self, m, alpha_i, lmbda):

        d = (1/500)*1e-3
        m = torch.unsqueeze(m, dim=1)
        lmbda = torch.unsqueeze(lmbda, dim = 0)
        alpha_i = alpha_i.unsqueeze(dim = 0)
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
    argument = Argument()
    arg = argument.parse()

    opt_param = torch.tensor([1.5, 0.5, 0.033,0.0005], dtype = torch.float, requires_grad = True, device = device)    

    lr = 1e-2
    decay_step = 1000

    epoch = 2000
    loss_f = torch.nn.L1Loss()
    losses = []
    
    optimizer = torch.optim.Adam([opt_param], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.9)

    pixel_num = 6
    
    for i in range(epoch):        
        depth = torch.zeros(size=(pixel_num,), device = device)

        depth[0], depth[1], depth[2], depth[3], depth[4], depth[5] = 1.47922784,1.47858544,1.47261183,1.49140792,1.53930147,1.58390314
        renderer = PixelRenderer(arg=arg, opt_param= opt_param, depth = depth)
        uv_cam_m1, uv_cam_m_1, xy_wvl_m1_real, xy_wvl_m_1_real = renderer.render(depth.unsqueeze(dim = 0), i)

        loss_m1 = loss_f(uv_cam_m1.to(torch.float32),xy_wvl_m1_real.to(torch.float32))
        loss_m_1 = loss_f(uv_cam_m_1.to(torch.float32), xy_wvl_m_1_real.to(torch.float32))
        
        loss = (loss_m1 + loss_m_1)
        
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        
        scheduler.step()

        if i % 100 == 0:
            print(f" Opt param value : {opt_param}, Epoch : {i}/{epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
            print(renderer.extrinsic_diff.detach().cpu().numpy())
    plt.figure()
    plt.plot(losses)
    plt.savefig('./loss_ftn.png')
