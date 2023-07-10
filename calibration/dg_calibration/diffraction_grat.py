import torch
import sys, os, cv2
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils.ArgParser import Argument

from hyper_sl.image_formation.projector import Projector
from hyper_sl.image_formation.camera import Camera

from hyper_sl.utils import calibrated_params

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from calibration.dg_calibration import point_process

class PixelRenderer():
    """ Render for a single scene 
        which has specific # of pixels for N different patterns
        
    """
    def __init__(self, arg, opt_param, illum_dir, n_pattern, grid_pts):
        self.arg = arg
        
        # device
        self.device = arg.device
        
        # arguments
        self.wvls = arg.wvls
        self.n_illum = 1
        self.opt_param = opt_param
        self.illum_dir = illum_dir
        self.n_pattern = n_pattern
        self.grid_pts = grid_pts
        
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
        self.proj_focal_length = self.proj.focal_length_proj()
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        
        # m order, wvls
        self.m_list = arg.m_list.to(device=self.device)
        self.m_n = len(self.m_list)
        self.wvls = torch.tensor([450,500,550,600,650])*1e-9
        self.wvls = self.wvls.to(device=self.device)
        self.wvls_n = len(self.wvls)
        
        # make sph2cart
        self.u = self.sph2cart(opt_param[0], opt_param[1])

        self.extrinsic_diff = self.ext_diff(self.u, opt_param[2], opt_param[3])
        
        # projector sensor plane
        self.xyz1, self.proj_center = self.proj_sensor_plane(illum_dir= self.illum_dir, n_pattern= self.n_pattern)
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
        extrinsic_diff = torch.zeros((4,4), device= self.device)
        
        self.rot_mat = self.rotation_matrix(u, k)

        # rotation
        extrinsic_diff[:3,:3] = self.rot_mat

        # translate
        extrinsic_diff[0,3] = 0.
        extrinsic_diff[1,3] = 0.
        extrinsic_diff[2,3] = point1
        extrinsic_diff[3,3] = 1
                
        return extrinsic_diff

    def render(self, depth_dir, processed_points):        

        self.XYZ_p = self.unproj_proj()    
        self.XYZ_c = self.unproj_cam()
        
        depth = self.get_depth(self.arg, depth_dir, processed_points)
                
        # constant where z equals to depth value
        t = (depth-self.intersection_points_proj[2])/self.z
        
        # 3D XYZ points
        self.X, self.Y, self.Z = self.intersection_points_proj[0] + self.alpha_m*t, self.intersection_points_proj[1] + self.beta_m*t, self.intersection_points_proj[2] + self.z*t
        
        ##### CAM COORDINATE
        # project XYZ proj coord onto cam plane / uv : cam H, W / xy : real cam coord
        uv_cam = self.projection(self.X,self.Y,self.Z)

        # predicted uv cam coords
        uv_cam = uv_cam.reshape(3, self.m_n, self.wvls_n, self.pixel_num)[:2]
    
        processed_points = processed_points.permute(3,0,1,2)
        
        return uv_cam, processed_points
    
    def distortion(self, XYZ_cam, cam_dist):
        
        k1, k2, p1, p2, k3 = cam_dist[:,0], cam_dist[:,1], cam_dist[:,2], cam_dist[:,3], cam_dist[:,4]
        X_c, Y_c, Z_c = XYZ_cam[0], XYZ_cam[1], XYZ_cam[2]
        
        x_prime, y_prime = X_c / Z_c, Y_c / Z_c
        r = torch.sqrt(x_prime**2 + y_prime**2)        
        
        x_distorted = X_c * (1 + k1*r**2 + k2 * r**4 + k3 * r**6) + (2*p1*X_c*Y_c + p2*(r**2 + 2*X_c**2))
        y_distorted = Y_c * (1 + k1*r**2 + k2 * r**4 + k3 * r**6) + (p1*(r**2 + 2*Y_c**2) + 2*p2*X_c*Y_c)
        
        XYZ_dist = torch.stack((x_distorted, y_distorted, XYZ_cam[2,:]), dim = 0) # m, wvl, xyz1, px

        return XYZ_dist
    
    def unproj_cam(self):
        uv1_c = torch.tensor([234., 285., 1.], device=self.device)
        
        z_val = 645.0721 * 1e-3
        suv_c = uv1_c * z_val
        
        XYZ_c = (torch.linalg.inv(self.int_cam).to(self.device)@suv_c)        
        
        return XYZ_c
        
    def unproj_proj(self):
        uv_p = self.get_grid_points_undistort(self.illum_dir, self.n_pattern).squeeze()
        ones = torch.ones(size = (uv_p.shape[0],1), device = self.device)
        uv1_p = torch.hstack((uv_p, ones)).T
        
        z_val = 0.7555901
        suv_p = uv1_p * z_val
        
        XYZ_p = (torch.linalg.inv(self.int_proj).to(self.device)@suv_p)
        
        return XYZ_p
        
    def visualization(self, uv_cam, real_uv, n, i):
        
        fig, ax = plt.subplots(figsize = (10,5))
        plt.plot(real_uv[0].detach().cpu().numpy().flatten(), real_uv[1].detach().cpu().numpy().flatten(), '.', label = 'gt')
        plt.plot(uv_cam[0].detach().cpu().numpy().flatten(), uv_cam[1].detach().cpu().numpy().flatten(), '.', label = 'pred')

        plt.axis('equal')
        plt.xlim(-10,900)
        plt.ylim(-10,600)
        rect = patches.Rectangle((0,0), 890,580, linewidth = 1, edgecolor = 'r', facecolor='none')
        ax.add_patch(rect)
        ax.legend(loc = 'best')
        ax.invert_yaxis()
        plt.savefig('./dg_cal/real_and_uv_%03d_patt_%06d_epoch.png' %(n, i))
    
    def projection(self, X, Y, Z):
        """
            proj coord to world/cam coord
        """

        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        XYZ1 = torch.stack((X,Y,Z,torch.ones_like(X)), dim = 0)
        
        # XYZ 3D points proj coord -> cam coord                   
        XYZ_cam = (self.ext_proj)@XYZ1
        
        # XYZ_cam reshape 임의로 1 order, 0번째 pixel grid, z 좌표를 spectralon depth로 고침
        XYZ_cam = XYZ_cam.reshape(4, self.m_n, self.wvls_n, self.grid_pts)
        # XYZ_cam[:, 1, :, 0] = torch.tensor([-0.05250806, 0.07126567, 0.76639038 , 1.]).unsqueeze(dim = 1) # patt 4, grid 2, depth xyz / 227 283 (x, y)

        XYZ_cam = XYZ_cam.reshape(4, -1)
        
        # distortion
        cam_dist = torch.tensor(self.undistort_cam(), device=self.device).type(torch.float32)
        XYZ_dist = self.distortion(XYZ_cam, cam_dist)

        # Opencv project points
        # cam_int, cam_dist = calibrated_params.bring_params(arg.calibration_param_path, "cam")

        # r_mat = np.eye(3)
        # t_mat = np.array([[0.],[0.],[0.]])
        
        # XYZ_cam = XYZ_cam.detach().cpu().numpy().transpose(1, 0)[:,:3]
        # # dmap_SL_reshape = torch.stack((self.X, self.Y, self.Z), axis = 3).reshape(-1, 3)
        # # dmap_SL_reshape = dmap_SL_reshape.detach().cpu().numpy()
        # uv_cam = cv2.projectPoints(XYZ_cam, r_mat, t_mat, cam_int, cam_dist)

        # uv cam coord
        uv_cam = (self.int_cam.to(self.device))@XYZ_dist
        uv_cam = uv_cam / uv_cam[2]
        
        return uv_cam
    
    def proj_sensor_plane(self, illum_dir, n_pattern):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """
        uv_p = self.get_grid_points_undistort(illum_dir, n_pattern).squeeze()
        ones = torch.ones(size = (uv_p.shape[0],1), device = self.device)
        uv1_p = torch.hstack((uv_p, ones)).T
        
        # uv1_p : x, y, 1 순으로 pixel num 만큼
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
    
    def get_depth(self, arg, depth_dir, processed_points):
        """
            depth values for each points
        """
        
        # depth values in world coordinate
        depth = torch.tensor(np.load(depth_dir)).type(torch.float32).to(self.device)
        ones = torch.ones(size = (arg.cam_H, arg.cam_W, 1), device = self.device)
        depth1 = torch.concat((depth, ones), dim = 2)
        
        # depth values in projector coordinate
        depth_proj = torch.linalg.inv(self.ext_proj)@depth1.permute(2,0,1).reshape(4,-1)
        depth_proj = depth_proj.reshape(-1, arg.cam_H, arg.cam_W)
        depth_proj = depth_proj[2]

        # pick depth for 0-order, -1 order, 1 order
        # 3, 5, 6
        depth = torch.zeros(size=(self.m_n, self.wvls_n, self.pixel_num), device= self.device)
        depth = depth.reshape(-1, self.pixel_num)
        
        processed_points = processed_points.reshape(-1, self.pixel_num, 2)
        
        for i in range(self.pixel_num):
            depth[...,i] = depth_proj[processed_points[...,i,1], processed_points[...,i,0]]
        
        depth = depth.to(self.device).reshape(self.m_n, self.wvls_n, self.pixel_num)

        return depth
    
    # get illumination grid 0-order points
    def get_grid_points_undistort(self, illum_dir, n_pattern):
        """
            get illumination grid 0-order points (undistorted points)
        """
        
        proj_int, proj_dist, _, _ = calibrated_params.bring_params(arg.calibration_param_path, "proj")

        illum_grid_points = np.load(os.path.join(illum_dir, "pattern_%03d.npy" %n_pattern)).reshape(-1,1,2).astype(np.float32)
        illum_grid_points_undistort = torch.tensor(cv2.undistortPoints(illum_grid_points, proj_int, proj_dist, P=proj_int), device= self.device)
        
        return illum_grid_points_undistort

    def undistort_cam(self):
        _, cam_dist = calibrated_params.bring_params(arg.calibration_param_path, "cam")
        return cam_dist
   
if __name__ == "__main__":    
    argument = Argument()
    arg = argument.parse()

    # date
    date = 'test_2023_07_09_15_37'
    
    # grid points
    grid_pts = 5
    
    # depth dir
    depth_dir = "./calibration/gray_code_depth/spectralon_depth_0708.npy"
    
    # illum dir 
    illum_dir = './calibration/dg_calibration/' + date + '_patterns'
    total_dir = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/"
    point_dir = total_dir + date + '_points'
    
    N_pattern = len(os.listdir(point_dir))
    wvls = np.arange(450, 660, 50)
    
    # optimized param initial values
    # initial_value = torch.tensor([ 1., 1., 0.5, 0.03])
    initial_value = torch.tensor([ 1.5, 1., 0.8, 0.01])
    
    # load opt param
    # initial_value = torch.tensor(np.load('./calibration/dg_calibration/dg_extrinsic/opt_param_single_%s_%06d.npy' %(date, 540)))
    
    # parameters to be optimized
    opt_param = torch.tensor(initial_value, dtype= torch.float, requires_grad=True, device= arg.device)
    
    # loss ftn
    loss_f = torch.nn.L1Loss()
    losses = []
    
    test = False
    
    ### TESTING FOR ZERO ORDER / PATTERN 3
    if test == True:
        # Testing pattern 3
        for n in range(0,1):
            renderer = PixelRenderer(arg=arg, opt_param= opt_param, illum_dir = illum_dir, n_pattern= n, grid_pts= grid_pts)
            
            pattern_dir = point_dir + '/pattern_%02d'%n
            processed_points = torch.tensor(point_process.point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, n),device=arg.device, dtype= torch.long)
            
            # 몇번째 pattern인지를 넣어주면 unproj & proj 한 uv coordinate output
            uv_cam, real_uv = renderer.render(depth_dir= depth_dir, processed_points = processed_points)
            renderer.visualization(uv_cam, real_uv, n, 0)
            
            # first pattern, loss 
            if n == 0:
                # only first orders
                uv_cam, real_uv = torch.stack((uv_cam[:,0], uv_cam[:,2]), dim =1), torch.stack((real_uv[:,0], real_uv[:,2]), dim =1)
                
                # valid
                uv_cam_flatten, real_uv_flatten = uv_cam.flatten(), real_uv.flatten()
                check = real_uv_flatten != 0
                uv_cam_valid, real_uv_valid = uv_cam_flatten[check], real_uv_flatten[check]
                
                # loss
                loss_patt = loss_f(uv_cam_valid.to(torch.float32), real_uv_valid.to(torch.float32))
                loss = loss_patt
                
            else:
                # only first orders
                uv_cam, real_uv = torch.stack((uv_cam[:,0], uv_cam[:,2]), dim =1), torch.stack((real_uv[:,0], real_uv[:,2]), dim =1)
                
                # valid
                uv_cam_flatten, real_uv_flatten = uv_cam.flatten(), real_uv.flatten()
                check = real_uv_flatten != 0
                uv_cam_valid, real_uv_valid = uv_cam_flatten[check], real_uv_flatten[check]
                
                # loss
                loss_patt = loss_f(uv_cam_valid.to(torch.float32), real_uv_valid.to(torch.float32))
                loss = loss + loss_patt
    
    else: 
        # training args
        lr = 1e-2 # 1e-3
        decay_step = 250 # 1000
        epoch = 1500
        
        optimizer = torch.optim.Adam([opt_param], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.8)
                
        for i in range(epoch):        
            for n in range(N_pattern):
                renderer = PixelRenderer(arg=arg, opt_param= opt_param, illum_dir = illum_dir, n_pattern= n, grid_pts= grid_pts)
                
                pattern_dir = point_dir + '/pattern_%02d'%n
                processed_points = torch.tensor(point_process.point_process(arg, grid_pts, total_dir, date, pattern_dir, wvls, n),device=arg.device, dtype= torch.long)
                
                # 몇번째 pattern인지를 넣어주면 unproj & proj 한 uv coordinate output
                uv_cam, real_uv = renderer.render(depth_dir= depth_dir, processed_points = processed_points)
                if i % 30 == 0 :
                    renderer.visualization(uv_cam, real_uv, n, i)
                
                # first pattern, loss 
                if n == 0:
                    # only first orders
                    uv_cam, real_uv = torch.stack((uv_cam[:,0], uv_cam[:,2]), dim =1), torch.stack((real_uv[:,0], real_uv[:,2]), dim =1)
                    
                    # valid
                    uv_cam_flatten, real_uv_flatten = uv_cam.flatten(), real_uv.flatten()
                    check = real_uv_flatten != 0
                    uv_cam_valid, real_uv_valid = uv_cam_flatten[check], real_uv_flatten[check]
                    
                    # loss
                    loss_patt = loss_f(uv_cam_valid.to(torch.float32), real_uv_valid.to(torch.float32))
                    loss = loss_patt
                    
                else:
                    # only first orders
                    uv_cam, real_uv = torch.stack((uv_cam[:,0], uv_cam[:,2]), dim =1), torch.stack((real_uv[:,0], real_uv[:,2]), dim =1)
                    
                    # valid
                    uv_cam_flatten, real_uv_flatten = uv_cam.flatten(), real_uv.flatten()
                    check = real_uv_flatten != 0
                    uv_cam_valid, real_uv_valid = uv_cam_flatten[check], real_uv_flatten[check]
                    
                    # loss
                    loss_patt = loss_f(uv_cam_valid.to(torch.float32), real_uv_valid.to(torch.float32))
                    loss = loss + loss_patt
                    # loss = loss_patt

                # if i % 30 == 0:
                    # renderer.visualization(uv_cam, real_uv, n, i)
                    
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item() / N_pattern)
            optimizer.step()
            scheduler.step()

            if i % 30 == 0:
                print(f" Opt param value : {opt_param}, Epoch : {i}/{epoch}, Loss: {loss.item() / N_pattern}, LR: {optimizer.param_groups[0]['lr']}")
                print(renderer.extrinsic_diff.detach().cpu().numpy())
                
                np.save('./calibration/dg_calibration/dg_extrinsic/dg_extrinsic_single_%s_%06d' %(date, i), renderer.extrinsic_diff.detach().cpu().numpy())
                np.save('./calibration/dg_calibration/dg_extrinsic/opt_param_single_%s_%06d' %(date, i), opt_param.detach().cpu().numpy())
                
                plt.figure()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(losses)
                plt.savefig('./dg_cal/loss/loss_ftn_%06d.png'%i)
