import numpy as np
import torch, os, sys, cv2
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation.projector import Projector
from hyper_sl.image_formation.camera import Camera
from hyper_sl.image_formation import distortion
import torchvision.transforms as tf

from hyper_sl.data.create_data import createData

import time, math

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
        
        self.load_data = load_data.load_data(arg)
        self.noise = noise.GaussianNoise(0, arg.noise_std, arg.device)
        self.normalize = normalize
        self.create_data = createData
        
        # cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.cam_sensor_width = arg.sensor_width*1e-3
        self.cam_focal_length = arg.focal_length*1e-3
        self.cam_pitch = arg.cam_pitch
        self.CRF_cam = torch.tensor(self.cam.get_CRF()).to(self.device)
        
        # proj
        self.proj_sensor_diag = arg.sensor_diag_proj *1e-3
        self.proj_focal_length = arg.focal_length_proj *1e-3
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.sensor_height_proj = (torch.sin(torch.atan2(torch.tensor(self.proj_H),torch.tensor(self.proj_W)))*self.proj_sensor_diag)
        self.proj_pitch = (self.sensor_height_proj/ (self.proj_H))
        self.intrinsic_proj_real = self.proj.intrinsic_proj_real()
        self.CRF_proj = torch.tensor(self.proj.get_PRF()).to(self.device)

        # dg intensity
        self.dg_intensity = torch.tensor(self.proj.get_dg_intensity(), device=self.device).unsqueeze(dim=0).float()
        
        # gaussian blur
        self.gaussian_blur = tf.GaussianBlur(kernel_size=(11,11), sigma=(3.5,3.5))
        
        # path
        self.dat_dir = arg.dat_dir
        # self.result_dir = arg.render_result_dir
        
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
        self.alpha_i = self.incident_dir_unit[0,:]
        self.beta_i = self.incident_dir_unit[1,:]

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
        self.intersection_points_r = self.intersection_points.reshape(self.m_n,self.proj_H*self.proj_W) 
        self.diffracted_dir_unit_r = self.diffracted_dir_unit.reshape(self.m_n,self.m_n, self.wvls_n, self.proj_H*self.proj_W)

        # optical center
        self.p = self.intersection_points_r.T
        self.d = self.diffracted_dir_unit_r.T
        
        # finding point p
        self.optical_center_virtual = self.proj.get_virtual_center(self.p,self.d, self.wvls_n, self.m_n)
        
        # optical_center_virtual shape : m_N, wvls_N, 3
        self.optical_center_virtual = torch.tensor(self.optical_center_virtual, dtype=torch.float32, device= self.device) 
        self.optical_center_proj = self.proj_center_dg.to(self.device) 

        # distortion coefficient
        self.p_list = self.dist.bring_distortion_coeff(arg, self.m_list, self.wvls, self.dat_dir)
        self.p_list = self.p_list.to(device=self.device)

    def render(self, depth, normal, hyp, occ, cam_coord, eval, illum_only=False):
        """
            input   
                depth : B, # pixel
                normal : : B, 3(xyz), # pixel
                hyp : B, # pixel, 25
                occ : B, # pixel
                cam_coord : B, # pixel, 2 (xy)
                illum_opt : optimized illumination
                illum_only : outputs only illumination data
        
        """
        print('rendering start')
        render_start = time.time()
        math.factorial(100000)
        
        self.batch_size = depth.shape[0]
        self.pixel_num = depth.shape[1]

        illum_data = torch.zeros(size =(self.batch_size, self.pixel_num, self.n_illum, self.wvls_n))
        
        if not illum_only:
            occ = occ.to(self.device).unsqueeze(dim = 1)
            hyp = torch.tensor(hyp, device= self.device)
            
            normal = normal.to(self.device)
            normal_vec_unit_clip = normal
    
        # depth2XYZ
        X, Y, Z = self.cam.unprojection(depth = depth, cam_coord = cam_coord)

        # change XYZ to dg coord
        XYZ_dg = self.proj.XYZ_to_dg(X,Y,Z)

        # shading term
        # B, m, 29, 3, # pixel
        illum_vec_unit = self.illum_unit(X,Y,Z)
        
        if not illum_only:
            shading = (illum_vec_unit*(normal_vec_unit_clip[:,None,:,:].unsqueeze(dim = 1))).sum(axis = 3).squeeze()
            shading = shading.repeat(self.batch_size, self.m_n, self.wvls_n, 1)
            
        # find the intersection points with dg and the line XYZ-virtual proj optical center in dg coordinate
        intersection_points_dg = self.proj.intersections_dg(self.optical_center_virtual, XYZ_dg)
        # distortion function
        sensor_X_virtual_distorted, sensor_Y_virtual_distorted = self.dist.distort_func(intersection_points_dg[..., 0,:]/self.proj_pitch, intersection_points_dg[...,1,:]/self.proj_pitch, self.p_list[...,0,:], self.p_list[...,1,:])
        sensor_X_virtual_distorted, sensor_Y_virtual_distorted = sensor_X_virtual_distorted*self.proj_pitch, sensor_Y_virtual_distorted*self.proj_pitch
        
        # dg intersection pts to proj coords and connect with proj center and find the intersection pts at 0.008
        intersection_points_dg_real1 = torch.stack((sensor_X_virtual_distorted, sensor_Y_virtual_distorted, torch.zeros_like(sensor_X_virtual_distorted), torch.ones_like(sensor_X_virtual_distorted)), dim = 3)
        intersection_points_proj_real = self.proj.intersect_points_to_proj(intersection_points_dg_real1)
        
        # find proj's xy coords
        xyz_proj = self.proj.projection(intersection_points_proj_real)
        xy_proj_real_norm = self.normalize.normalization(self.arg, xyz_proj[:,1,0,:2,...].permute(0,2,1))

        # xy_proj to uv
        uv1 = self.proj.xy_to_uv(xyz_proj)

        new_idx, cond = self.get_newidx(uv1)
        
        cam_N_img = torch.zeros(size=(self.batch_size, self.pixel_num, self.n_illum, 3), device= self.device)
        
        for j in range(self.n_illum):
            illum = self.load_data.load_illum(j).to(self.device) 
            illum_img = torch.zeros(self.batch_size, self.m_n, self.wvls_n, self.pixel_num, device= self.device).flatten()

            illum_hyp = illum.reshape((self.proj_H*self.proj_W, 3))@((self.CRF_proj.T).type(torch.float32))
            illum_hyp = illum_hyp.reshape((self.proj_H,self.proj_W,self.wvls_n))
            illum_hyp_unsq = illum_hyp.repeat(3,1,1,1)
            illum_hyp_unsq = illum_hyp_unsq.permute(0,3,1,2)
            illum_hyp_unsq = illum_hyp_unsq.repeat(self.batch_size, 1,1,1,1)

            hyp_f = illum_hyp_unsq.flatten()
            
            valid_pattern_img = hyp_f[new_idx]
            
            illum_img[cond.flatten()] = valid_pattern_img.flatten()
            
            illum_img = illum_img.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)
            illum_img = 0.2 * illum_img * self.dg_intensity.unsqueeze(dim=3)
            illums_m_img = illum_img.sum(axis = 1).reshape(self.batch_size, self.wvls_n, self.pixel_num).permute(0,2,1)
            
            if not illum_only:
                # multipy with occlusion
                illum_w_occ = illum_img*occ.unsqueeze(dim=1)
                illums_w_occ = illum_w_occ*shading 

                illums_w_occ = illums_w_occ.permute(0,1,3,2)
                
                cam_m_img = torch.zeros((self.batch_size, self.m_n, self.pixel_num, 3))
                
                # m order에 따른 cam img : cam_m_img
                for k in range(self.m_n): 
                    cam_m_img[:,k,...] = (hyp* (illums_w_occ[:,k,...])@ self.CRF_cam)

                cam_img = cam_m_img.sum(axis=1)
            
                # # gaussian blur
                # if eval == True:
                #     cam_img = cam_img.reshape(-1, self.cam_H, self.cam_W, 3).permute(0,3,1,2)
                #     cam_N_img[...,j,:] = torch.clamp(self.gaussian_blur(cam_img), 0, 1).permute(0,2,3,1).reshape(-1, self.pixel_num, 3)
                # else:
                #     cam_N_img[...,j,:] = torch.clamp(cam_img, 0, 1)
            
            cam_N_img[...,j,:] = cam_img
            illum_data[:,:,j,:] = illums_m_img

        if illum_only:
            return None, xy_proj_real_norm, illum_data, None

        # noise
        # if eval == False:
        noise = self.noise.sample(cam_N_img.shape)
        cam_N_img += noise
        cam_N_img = torch.clamp(cam_N_img, 0, 1)
        
        render_end = time.time()
        
        print(f"render time : {render_end - render_start:.5f} sec")
        print(f'rendering finished for iteration')
                
        return cam_N_img, xy_proj_real_norm, illum_data, shading
    
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
        a = t.unsqueeze(dim = 1).T * incident_dir_unit + proj_center_dg
        
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
    
    def illum_unit(self, X,Y,Z):
        """ 
        inputs : X,Y,Z world coord
                virtual projector center in dg coord (optical_center_virtual) # need to change it to world coord
        """
        XYZ = torch.stack((X,Y,Z), dim = 1).to(self.device)
        XYZ = XYZ.unsqueeze(dim=1)
        
        # optical_center_proj
        optical_center_proj = torch.tensor([[0.],[0.],[0.],[1.]], device=self.device)

        optical_center_world = self.extrinsic_proj_real@optical_center_proj
        optical_center_world = optical_center_world[:3] # m_N, wvls_N, 3
        optical_center_world = optical_center_world.unsqueeze(dim = 0)
        
        # illumination vector in world coord
        illum_vec =  optical_center_world.unsqueeze(dim =0) - XYZ

        illum_norm = torch.norm(illum_vec, dim = 2) # dim = 0
        illum_norm = torch.unsqueeze(illum_norm, dim = 2)
        
        illum_unit = illum_vec/illum_norm
        illum_unit = illum_unit.unsqueeze(dim =0)
        
        return illum_unit
    
    def get_newidx(self, uv1):
        
        r_proj, c_proj = uv1[:,:,:,1], uv1[:,:,:,0]
        cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W) 
        
        r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]
        r_proj_valid, c_proj_valid = torch.tensor(r_proj_valid), torch.tensor(c_proj_valid)  # TODO: do we need this? 

        batch_samples = torch.linspace(0, self.batch_size-1, self.batch_size,device=self.device)
        wvl_samples = torch.linspace(0, self.wvls_n-1, self.wvls_n,device=self.device)
        m_samples = torch.linspace(0, self.m_n-1, self.m_n,device=self.device)

        pixel_samples = torch.linspace(0, self.pixel_num-1, self.pixel_num, device= self.device)
        grid_b, grid_m, grid_w, grid_pixel = torch.meshgrid(batch_samples,m_samples,wvl_samples,pixel_samples,indexing='ij')

        grid_b_valid = grid_b.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
        grid_m_valid = grid_m.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
        grid_w_valid = grid_w.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]

        new_idx = self.m_n * self.wvls_n * self.proj_H * self.proj_W * grid_b_valid.long() \
                + self.wvls_n * self.proj_H * self.proj_W * grid_m_valid.long() \
                + self.proj_H * self.proj_W * grid_w_valid.long() \
                + self.proj_W * r_proj_valid.long() \
                + c_proj_valid.long()
        
        return new_idx, cond
    
if __name__ == "__main__":
   
    import torch, os, sys
    sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
    from scipy.io import loadmat
    from hyper_sl.utils.ArgParser import Argument
    from hyper_sl.data import create_data_patch
    from hyper_sl.utils import data_process
    
    argument = Argument()
    arg = argument.parse()
    
    # 기존의 hyperpsectral 정보와 depth로 rendering 하는 코드
    create_data = create_data_patch.createData
    
    plane_XYZ = torch.tensor(loadmat('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/hyper_sl/image_formation/rendering_prac/plane_XYZ.mat')['XYZ_q'])
    plane_XYZ = data_process.crop(plane_XYZ)
    
    
    pixel_num = arg.cam_H * arg.cam_W
    random = False
    index = 0
    
    depth = create_data(arg, "depth", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    # depth = torch.ones_like(depth)
    # depth[:] = plane_XYZ.reshape(-1,3)[:,2].unsqueeze(dim =0)*1e-3
    
    normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    # normal = torch.ones_like(normal)
    
    hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    # hyp = torch.ones_like(hyp)
    
    occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    # occ = torch.ones_like(occ)
    
    cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0)
    
    import cv2
    illum = cv2.imread("C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/illum/grid.png").astype(np.float32)
    # illum = cv2.imread("C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/illum/line_pattern.png").astype(np.float32)
    # illum = cv2.imread("C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/hyper_sl/image_formation/rendering_prac/MicrosoftTeams-image (11).png").astype(np.float32)
    # illum = cv2.imread("C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/illum/graycode_pattern/pattern_38.png").astype(np.float32)
    illum = cv2.cvtColor(illum, cv2.COLOR_BGR2RGB)
    illum = illum / 255.
    illum = torch.tensor(illum, device='cuda').unsqueeze(dim = 0)

    # n_scene, random, pixel_num, eval
    cam_N_img, xy_proj_real_norm, illum_data, shading = PixelRenderer(arg).render(depth = depth, normal = normal, hyp = hyp, cam_coord = cam_coord, occ = occ, eval = True)
    
    print('end')