import torch
import sys
# sys.path.append('C:/Users/mainuser/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')
sys.path.append('/workspace/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument

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
    def __init__(self, arg, opt_param):
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
                
        # cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.cam_focal_length = arg.focal_length*1e-3
        
        # int & ext
        self.int_cam = self.cam.intrinsic_cam()
        self.ext_proj = self.proj.extrinsic_proj_real()
        self.int_proj = self.proj.intrinsic_proj_real()
        
        # proj
        self.proj_focal_length = arg.focal_length_proj *1e-3
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.proj_pitch = 7.9559e-06
        
        # m order, wvls
        self.m_list = arg.m_list.to(device=self.device)
        self.m_n = len(self.m_list)
        self.wvls = torch.tensor([450,500,550,600,650])*1e-9
        self.wvls = self.wvls.to(device=device)
        self.wvls_n = len(self.wvls)
        
        # make sph2cart
        self.u = self.sph2cart(opt_param[0], opt_param[1])

        self.extrinsic_diff = self.ext_diff(self.u, opt_param[2], opt_param[3])

        # projector sensor plane
        self.xyz1, self.proj_center = self.proj_sensor_plane()
        self.pixel_num = self.xyz1.shape[1]
        
        # change proj sensor plane to dg coord
        self.xyz_proj_dg = self.extrinsic_diff@self.xyz1
        self.xyz_proj_dg = self.xyz_proj_dg[:3]

        # change proj center to dg coord
        self.proj_center_dg = (self.extrinsic_diff)@self.proj_center
        self.proj_center_dg = self.proj_center_dg[:3]
        
        # incident light, intersection points in DG coord
        # incident light dir vectors
        self.incident_dir = self.xyz_proj_dg - self.proj_center_dg
        # incident light dir in proj coord
        self.incident_dir_proj = self.xyz1[:3] - self.proj_center[:3]
        
        # make incident dir to unit vector
        self.norm = torch.norm(self.incident_dir, dim = 0)
        # make incident dir to unit vector in proj coord
        self.norm_proj = torch.norm(self.incident_dir_proj, dim = 0)
        
        # incident light unit dir vector
        self.incident_dir_unit = self.incident_dir/self.norm
        # proj coord
        self.incident_dir_unit_proj = self.incident_dir_proj/self.norm_proj
        
        # intersection points in dg coord
        self.intersection_points = self.find_intersection(self.proj_center_dg,self.incident_dir_unit)
        
        # intersection points in dg coord to proj coord
        self.ones = torch.ones(size=(self.intersection_points.shape[1],), device = self.device).unsqueeze(dim = 0)
        self.intersection_points1 = torch.concat((self.intersection_points, self.ones), dim = 0)
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
        x = r * torch.sin(elevation) * torch.cos(azimuth)
        y = r * torch.sin(elevation) * torch.sin(azimuth)
        z = r * torch.cos(elevation)
        
        u_vec = torch.stack((x,y,z), dim = 0)

        return u_vec
    
    def rotation_matrix(self, u, k):
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

    def ext_diff(self, u, k, point1): # [0],[1] 로 self.u 만들고, k = [3], point = [2]
        """
            diffraction grating extrinsic matrix (dg coord = inv(ext diff) @ proj plane coord)
            
            rotation matrix : normal vector of diffraction gratings & normal vector of projector plane
        
        """
        extrinsic_diff = torch.zeros((4,4), device= device)
        
        self.rot_mat = self.rotation_matrix(u, k)

        # rotation
        extrinsic_diff[:3,:3] = self.rot_mat

        # translate
        extrinsic_diff[0,3] = 0.
        extrinsic_diff[1,3] = 0.
        extrinsic_diff[2,3] = point1
        extrinsic_diff[3,3] = 1
                
        return extrinsic_diff


    def render(self, point_3d, i):        
        # depth for m = [-1,0,1], wvl
        depth = torch.tensor(np.load("./calibration/spectralon_depth.npy")).type(torch.float32).to(self.device)
        ones = torch.ones(size = (arg.cam_H, arg.cam_W, 1), device = self.device)
        depth1 = torch.concat((depth, ones), dim = 2)
        
        depth_proj = torch.linalg.inv(self.ext_proj)@depth1.permute(2,0,1).reshape(4,-1)
        depth_proj = depth_proj.reshape(-1, arg.cam_H, arg.cam_W)[2]
        
        depth = point_3d[2]
        depth = (depth).repeat(self.m_n, 1)
        depth = torch.unsqueeze(depth, 1).repeat(1, self.wvls_n, 1)
        depth = depth.reshape(self.m_n, self.wvls_n, self.pixel_num) 
        
        # m = 0 order
        depth[1,:,0], depth[1,:,1], depth[1,:,2], depth[1,:,3], depth[1,:,4], depth[1,:,5] = depth_proj[43,117], depth_proj[295,116],depth_proj[547,113],depth_proj[53,799],depth_proj[299,803],depth_proj[550,811]
        # m = -1 order
        depth[0,0,0], depth[0,1,0], depth[0,2,0], depth[0,3,0], depth[0,4,0] = depth_proj[54,500], depth_proj[53,538],depth_proj[53,582],depth_proj[53,622],depth_proj[53,662]
        depth[0,0,1], depth[0,1,1], depth[0,2,1], depth[0,3,1], depth[0,4,1] = depth_proj[298,496], depth_proj[299,537],depth_proj[299,579],depth_proj[299,619],depth_proj[299,660]
        depth[0,0,2], depth[0,1,2], depth[0,2,2], depth[0,3,2], depth[0,4,2] = depth_proj[546,496], depth_proj[546,536],depth_proj[547,579],depth_proj[547,620],depth_proj[547,661]
        # m = 1 order
        depth[2,0,3], depth[2,1,3], depth[2,2,3], depth[2,3,3], depth[2,4,3] = depth_proj[56,421], depth_proj[54,318],depth_proj[54,336],depth_proj[53,298],depth_proj[52,254]
        depth[2,0,4], depth[2,1,4], depth[2,2,4], depth[2,3,4], depth[2,4,4] = depth_proj[300,428], depth_proj[299,385],depth_proj[299,343],depth_proj[299,303],depth_proj[299,261]
        depth[2,0,5], depth[2,1,5], depth[2,2,5], depth[2,3,5], depth[2,4,5] = depth_proj[549,432], depth_proj[548,388],depth_proj[549,347],depth_proj[549,309],depth_proj[549,264]

        depth = depth.to(self.device)
        
        # constant where z equals to depth value
        t = (depth-self.intersection_points_proj[2])/self.z
        
        # 3D XYZ points
        self.X, self.Y, self.Z = self.intersection_points_proj[0] + self.alpha_m*t, self.intersection_points_proj[1] + self.beta_m*t, self.intersection_points_proj[2] + self.z*t

        ##### CAM COORDINATE
        # project XYZ proj coord onto cam plane / uv : cam H, W / xy : real cam coord coord
        uv_cam = self.projection(self.X,self.Y,self.Z)
        
        # predicted uv cam coords
        uv_cam = uv_cam.reshape(3,self.m_n, self.wvls_n, self.pixel_num)[:2]
        uv_cam_m_1, uv_cam_m1 = uv_cam[:,0], uv_cam[:,2]
        uv_cam_m_1, uv_cam_m1 = uv_cam_m_1[...,:3], uv_cam_m1[...,3:]
        
        # ground truth m = 1, m = -1 
        self.xy_wvl_m1_real = torch.tensor([[[421,428,432],  # 1, 2, 3
                                            [378,385,388],      # 5개 : wvl
                                            [336,343,347],      # 2개 : x & y
                                            [298,303,306],
                                            [254,261,264]],
                                        
                                            [[56,300,549],
                                            [54,299,548],
                                            [53,299,549],
                                            [53,299,549],
                                            [52,299,549]]
                                        ], device= device)

        self.xy_wvl_m_1_real = torch.tensor([[[500,496,496],   # 4, 5, 6
                                            [538,537,536],
                                            [582,579,579],
                                            [622,619,620],
                                            [662,660,661]],
                                            
                                        [[54,298,546],
                                            [53,299,546],
                                            [53,299,547],
                                            [53,299,547],
                                            [53,299,547]]
                                        ], device= device)
            
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
    
    def projection(self, X,Y,Z):
        """
            proj coord to world/cam coord
        """

        # focal_length_proj_virtual : 3, 77

        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 0)

        # XYZ 3D points proj coord -> cam coord                   
        XYZ_cam = (self.ext_proj)@XYZ1

        # uv cam coord
        uv_cam = (self.int_cam.to(device))@XYZ_cam[:3]
        uv_cam = uv_cam / uv_cam[2]
        
        return uv_cam
    

    def proj_sensor_plane(self):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """

        # uv1_p = torch.tensor([[130.,70.,1.],[130.,200.,1.],[130.,330.,1,],[530.,70.,1,],[530.,200.,1,],[530.,330.,1.]]).T.to(self.device)
        # uv1_p = torch.tensor([[127.95245,66.53236,1.],[129.21448,199.01979,1.],[129.63133,329.66168,1,],[531.285,67.02354,1,],[530.3117,199.23007,1,],[529.98004,329.7011,1.]]).T.to(self.device)
        
        uv1_p = torch.tensor([[106.12076,33.806595,1.] ,[109.02093, 183.79155,1. ], [109.58386 ,329.5976,1. ],[511.54556,35.970963,1.],[510.32538,184.17296 ,1.],[510.00385,329.75537,1. ]]).T.to(self.device)
        # uv1_p = torch.tensor( [[110,40,1.], [110,185,1.], [110,330,1.], [510,40,1.], [510,185,1.], [510,330,1.]]).T.to(self.device)

        suv_p = uv1_p * self.proj_focal_length
        
        # to real projector plane size
        xyz_p = (torch.linalg.inv(self.int_proj).to(self.device)@suv_p)

        # proj_center
        proj_center = torch.zeros(size=(4,1), device=self.device)
        proj_center[3,0] = 1.

        # make projector sensor xyz1 vector
        xyz1 = torch.concat((xyz_p, torch.ones(size=(xyz_p.shape[1],), device = self.device).unsqueeze(dim =0)), dim = 0)
        xyz1 = xyz1.to(self.device)
        
        return xyz1, proj_center
    
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

    opt_param = torch.tensor([ 1.5, 0.5,  0.0033,  0.], dtype= torch.float, requires_grad=True,device=device)
    # opt_param = torch.tensor([1.5, 0.5, 0.0033,0.0005], dtype = torch.float, requires_grad = True, device = device)    

    lr = 1e-3
    decay_step = 1000

    epoch = 5000
    loss_f = torch.nn.L1Loss()
    losses = []
    
    optimizer = torch.optim.Adam([opt_param], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.7)

    pixel_num = 6
    
    for i in range(epoch):        
        depth = torch.zeros(size=(pixel_num,), device = device)
        
        # pattern points들의 undistort되지 않은 값을 넣은 3d point
        # points_3d_proj = torch.tensor([[-0.3405097 , -0.3241511 , -0.32005197,  0.28541371,  0.29591602, 0.29865494],
        #                             [-0.47573565, -0.24074295, -0.03094228, -0.40558114, -0.22458692, -0.02986078],
        #                             [ 1.7127738 ,  1.62668781,  1.60451955,  1.45514261,  1.5135434 , 1.52946535],
        #                             [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ]])
        
        # pattern points들의 undistort된 값을 넣은 3d point
        # points_3d_proj = torch.tensor([[-0.3223618 , -0.31818153, -0.31746865,  0.28996819,  0.29708312, 0.29797115],
        #                                 [-0.44691891, -0.23587377, -0.03086009, -0.41151851, -0.2255306 , -0.02993264],
        #                                 [ 1.60136494,  1.58956494,  1.58836879,  1.47290781,  1.51815022, 1.52646141],
        #                                 [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  1.        ]]
        
        # # patterns points들의 undistort된 값을 넣은 3d point
        # points_3d_proj = torch.tensor([[-0.17683013, -0.17474294, -0.17257632,  0.13482537,  0.13334565, 0.13035166],
        #                                 [-0.24674278, -0.129139  , -0.01528223, -0.23571209, -0.12372854, -0.01387424],
        #                                 [ 0.79332279,  0.79347202,  0.78565196,  0.75963267,  0.75788606, 0.74357495],
        #                                 [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ]])
        
        # patterns points들의 undistort하지 않은 값을 넣은 3d point
        points_3d_proj = torch.tensor(([[-0.18424177, -0.17636287, -0.17320164,  0.13339871,  0.13300931,0.13044898],
                                        [-0.26054861, -0.13060869, -0.0152366 , -0.23355212, -0.12340132,-0.01383224],
                                        [ 0.84429842,  0.80491452,  0.79014072,  0.75496662,  0.75674377, 0.7439909 ],
                                        [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ]]))
        
        renderer = PixelRenderer(arg=arg, opt_param= opt_param)
        uv_cam_m1, uv_cam_m_1, xy_wvl_m1_real, xy_wvl_m_1_real = renderer.render(point_3d= points_3d_proj, i = i)

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
