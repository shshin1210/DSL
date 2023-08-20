import numpy as np
import torch, os, sys, cv2
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation_method2.projector import Projector
from hyper_sl.image_formation_method2.camera import Camera
from hyper_sl.image_formation_method2 import distortion
import torchvision.transforms as tf

from hyper_sl.data.create_data_patch import createData

import time
import matplotlib.pyplot as plt


class PixelRenderer():
    """ Render for a single scene 
        which has specific # of pixels for N different patterns
        
    """
    def __init__(self, arg):
        self.arg = arg

        # device
        self.device = arg.device
        
        # path
        self.dat_dir = arg.dat_dir # Need to be changed
        
        # m order, wvls
        self.n_illum = arg.illum_num
        self.m_list = arg.m_list.to(device=self.device)
        self.m_n = len(self.m_list)
        self.wvls = arg.wvls.to(device=self.device)
        self.wvls_n = len(self.wvls)
        self.depth_list = arg.depth_list.to(device = self.arg.device)
        
        # classes
        self.proj = Projector(arg, device= self.device)
        self.cam = Camera(arg)
        self.load_data = load_data.load_data(arg)
        self.dist = distortion.Distortion(arg, self.wvls, self.depth_list)
        
        self.load_data = load_data.load_data(arg)
        self.noise = noise.GaussianNoise(0, arg.noise_std, arg.device)
        self.normalize = normalize.Normalize(arg)
        self.create_data = createData
        
        # cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.cam_sensor_width = arg.sensor_width*1e-3
        self.cam_focal_length = arg.focal_length*1e-3
        self.cam_pitch = arg.cam_pitch
        self.CRF_cam = torch.tensor(self.cam.get_CRF()).to(self.device)
        
        # proj
        self.proj_sensor_diag = arg.sensor_diag_proj *1e-3
        self.proj_focal_length = self.proj.focal_length_proj()        
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.proj_pitch = arg.proj_pitch
        self.intrinsic_proj_real = self.proj.intrinsic_proj_real()
        self.CRF_proj = torch.tensor(self.proj.get_PRF()).to(self.device)

        # dg intensity
        self.dg_intensity = torch.tensor(self.proj.get_dg_intensity(), device=self.device).unsqueeze(dim=0).float()
        
        # gaussian blur
        self.gaussian_blur = tf.GaussianBlur(kernel_size=(11,11), sigma=(1.5,1.5))
        
        # extrinsic matrix
        self.extrinsic_proj_real = self.proj.extrinsic_proj_real()

        # distortion coefficient
        self.p_list = self.dist.bring_distortion_coeff(self.dat_dir)
        self.p_list = self.p_list.to(device=self.device) # 2, 25, 31, 21, 2 (m, wvl, depth, coeffs, xy)

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
        
        self.batch_size = depth.shape[0]
        self.pixel_num = depth.shape[1]

        illum_data = torch.zeros(size =(self.batch_size, self.pixel_num, self.n_illum, self.wvls_n))
        
        if not illum_only:
            occ = occ.to(self.device).unsqueeze(dim = 1)
            hyp = torch.tensor(hyp, device= self.device)
            
            normal = normal.to(self.device)
            normal_vec_unit_clip = normal
    
        # 3d points XYZ : shape 1, 580*890
        X, Y, Z = self.cam.unprojection(depth = depth, cam_coord = cam_coord)
        
        # shading term
        # B, m, 29, 3, # pixel
        illum_vec_unit = self.illum_unit(X,Y,Z)

        if not illum_only:
            shading = (illum_vec_unit*(normal_vec_unit_clip.unsqueeze(dim = 1))).sum(axis = 2).squeeze(dim = 1)
            shading = shading.unsqueeze(dim = 1).repeat(1,self.m_n,1) # extend to B, m, wvl, pixel_num
            shading = shading.unsqueeze(dim = 2).repeat(1,1,self.wvls_n,1)
        
        # distortion function : input x, y, p, q for different depth z values
        # shape : m (2), wvl, depth, newaxis, 2
        sensor_U_distorted, sensor_V_distorted = self.dist.distort_func(X/self.proj_pitch, Y/self.proj_pitch, Z, self.p_list[...,0], self.p_list[...,1], "cm")
        
        # zero order conventional projection & un-projection
        zero_uv1 = self.proj.projection(X, Y, Z)
        
        # concat to uv for three m orders
        uv1 = self.proj.make_uv1(sensor_U_distorted, sensor_V_distorted, zero_uv1)
        
        # 여기서 uv를 가지고 projector의 xy pixel로 바꿔서 normalization?
        xyz_proj = self.proj.get_xyz_proj(uv1)
        # normalization
        xy_proj_real_norm = self.normalize.normalization(xyz_proj[:,1,0,:2,...].permute(0,2,1))
        
        # new idx
        new_idx, cond = self.get_newidx(uv1)
        
        cam_N_img = torch.zeros(size=(self.batch_size, self.pixel_num, self.n_illum, 3), device= self.device)
        
        for j in range(self.n_illum):            
            # illum = cv2.imread('./hyper_sl/image_formation/rendering_prac/MicrosoftTeams-image (11).png', -1)
            # illum = torch.tensor(illum).to(device = self.arg.device).type(torch.float32)
            illum = self.load_data.load_illum(j).to(self.device) * self.arg.illum_weight
            illum = self.gaussian_blur(illum.permute(2,0,1)).permute(1,2,0)
            illum_img = torch.zeros(self.batch_size, self.m_n, self.wvls_n, self.pixel_num, device= self.device).flatten()

            illum_hyp = illum.reshape((self.proj_H*self.proj_W, 3))@((self.CRF_proj.T).type(torch.float32))
            illum_hyp = illum_hyp.reshape((self.proj_H,self.proj_W,self.wvls_n))
            illum_hyp_unsq = illum_hyp.repeat(3,1,1,1)
            illum_hyp_unsq = illum_hyp_unsq.permute(0,3,1,2)
            illum_hyp_unsq = illum_hyp_unsq.repeat(self.batch_size, 1,1,1,1)

            hyp_f = illum_hyp_unsq.flatten()
            
            # grid sampling
            # illum_img = self.grid_sample(uv1, hyp_f, illum_img)
            
            # No grid sampling
            valid_pattern_img = hyp_f[new_idx]
            
            illum_img[cond.flatten()] = valid_pattern_img.flatten()
            
            illum_img = illum_img.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)
            # ==================================== dg intensity ====================================
            illum_img =  illum_img * self.dg_intensity.unsqueeze(dim=3)
            illums_m_img = illum_img.sum(axis = 1).reshape(self.batch_size, self.wvls_n, self.pixel_num).permute(0,2,1)
            
            if not illum_only:
                # multipy with occlusion
                illum_w_occ = illum_img*occ.unsqueeze(dim=1)
                illums_w_occ = illum_w_occ*shading 

                illums_w_occ = illums_w_occ.permute(0,1,3,2)
                
                cam_m_img = torch.zeros((self.batch_size, self.m_n, self.pixel_num, 3))
                
                # m order에 따른 cam img : cam_m_img
                for k in range(self.m_n): 
                    cam_m_img[:,k,...] =  0.1 * (hyp* (illums_w_occ[:,k,...])@ self.CRF_cam)

                cam_img = cam_m_img.sum(axis=1)
                cam_N_img[...,j,:] = cam_img
                
            illum_data[:,:,j,:] = illums_m_img

        if illum_only:
            return None, xy_proj_real_norm, illum_data, None
        
        noise = self.noise.sample(cam_N_img.shape)
        cam_N_img += noise
        cam_N_img = torch.clamp(cam_N_img, 0, 1)
        
        render_end = time.time()
        
        print(f"render time : {render_end - render_start:.5f} sec")
                
        return cam_N_img, xy_proj_real_norm, illum_data, shading
    
    
    def get_newidx(self, uv1):
        
        r_proj, c_proj = uv1[:,:,:,1], uv1[:,:,:,0]
        cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W) 
        
        r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]
        r_proj_valid, c_proj_valid = torch.tensor(r_proj_valid), torch.tensor(c_proj_valid)

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
    
    def vis(self, data, num, vmin, vmax):
        illum_num = num
        max_images_per_column = 5
        num_columns = (illum_num + max_images_per_column - 1) // max_images_per_column
        plt.figure(figsize=(10, 3*num_columns))

        for c in range(num_columns):
            start_index = c * max_images_per_column
            end_index = min(start_index + max_images_per_column, illum_num)
            num_images = end_index - start_index
                    
            for i in range(num_images):
                plt.subplot(num_columns, num_images, i + c * num_images + 1)
                plt.imshow(data[:, :, i + start_index], vmin=vmin, vmax=vmax)
                plt.axis('off')
                plt.title(f"Image {i + start_index}")
                # cv2.imwrite('./spectralon/spectralon_real_%04d_img.png'%(i+start_index), data[:, :, i + start_index, ::-1]*255.)
                        
                if i + start_index == illum_num - 1:
                    plt.colorbar()

        plt.show()    
    
    def illum_unit(self, X,Y,Z):
        """ 
        inputs : X,Y,Z world coord
    
        """
        XYZ = torch.stack((X,Y,Z), dim = 1).to(self.device)
        XYZ = XYZ.unsqueeze(dim=1)
        
        # optical_center_proj
        optical_center_proj = torch.tensor([[0.],[0.],[0.],[1.]], device=self.device)
        
        optical_center_world = self.extrinsic_proj_real@optical_center_proj
        optical_center_world = optical_center_world[:3] # m_N, wvls_N, 3
        optical_center_world = optical_center_world.unsqueeze(dim = 0)
        
        # illumination vector in world coord
        illum_vec =  optical_center_world.unsqueeze(dim = 1) - XYZ

        illum_norm = torch.norm(illum_vec, dim = 2) # dim = 0
        illum_norm = torch.unsqueeze(illum_norm, dim = 2)
        
        illum_unit = illum_vec/illum_norm
        
        return illum_unit
    
    # def grid_sample(self, uv1, hyp_f, illum_img):
        
    #     uv1_reshape = uv1.reshape(self.batch_size, self.m_n, self.wvls_n, 3, self.cam_H, self.cam_W)
        
    #     # split integer and decimal
    #     uv1_integer = uv1_reshape.long()
    #     u_float = (uv1_reshape - uv1_integer)[:,:,:,1]
    #     v_float = (uv1_reshape - uv1_integer)[:,:,:,0]
                
    #     top_left_weight = (1-u_float) * (1-v_float)
    #     top_right_weight = u_float * (1-v_float)
    #     bott_left_weight = (1-u_float) * v_float
    #     bott_right_weight = u_float * v_float
        
    #     A, A_cond = self.get_newidx(uv1.long(), 'A')
    #     B, B_cond = self.get_newidx(uv1.long(), 'B')
    #     C, C_cond = self.get_newidx(uv1.long(), 'C')
    #     D, D_cond = self.get_newidx(uv1.long(), 'D')

    #     # 여기서 grid sampling?
    #     valid_pattern_img_A = hyp_f[A]
    #     valid_pattern_img_B = hyp_f[B]
    #     valid_pattern_img_C = hyp_f[C]
    #     valid_pattern_img_D = hyp_f[D]

    #     illum_img[A_cond.flatten()] = valid_pattern_img_A.flatten()
    #     illum_img_A = top_left_weight.flatten() * illum_img

    #     illum_img[B_cond.flatten()] = valid_pattern_img_B.flatten()
    #     illum_img_B = top_right_weight.flatten() * illum_img

    #     illum_img[C_cond.flatten()] = valid_pattern_img_C.flatten()
    #     illum_img_C = bott_left_weight.flatten() * illum_img

    #     illum_img[D_cond.flatten()] = valid_pattern_img_D.flatten()
    #     illum_img_D = bott_right_weight.flatten() * illum_img
        
    #     illum_img_final = illum_img_A + illum_img_B + illum_img_C + illum_img_D
    #     illum_img_final = illum_img_final.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)

    #     return illum_img_final
        
    # def get_newidx(self, uv1, corner):
        
    #     if corner == 'A':
    #             r_proj, c_proj = uv1[:,:,:,1], uv1[:,:,:,0]
    #             cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W) 
    #     elif corner == 'B':
    #             r_proj, c_proj = uv1[:,:,:,1], uv1[:,:,:,0]
    #             r_proj += 1
    #             cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W) 
    #     elif corner == 'C':
    #             r_proj, c_proj = uv1[:,:,:,1], uv1[:,:,:,0]
    #             c_proj += 1
    #             cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W) 
    #     else:
    #             r_proj, c_proj = uv1[:,:,:,1], uv1[:,:,:,0]
    #             r_proj += 1
    #             c_proj += 1
    #             cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W) 
        
    #     r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]

    #     batch_samples = torch.linspace(0, self.batch_size-1, self.batch_size,device=self.device)
    #     wvl_samples = torch.linspace(0, self.wvls_n-1, self.wvls_n,device=self.device)
    #     m_samples = torch.linspace(0, self.m_n-1, self.m_n,device=self.device)

    #     pixel_samples = torch.linspace(0, self.pixel_num-1, self.pixel_num, device= self.device)
    #     grid_b, grid_m, grid_w, grid_pixel = torch.meshgrid(batch_samples,m_samples,wvl_samples,pixel_samples,indexing='ij')

    #     grid_b_valid = grid_b.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
    #     grid_m_valid = grid_m.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
    #     grid_w_valid = grid_w.reshape(self.batch_size, self.m_n, self.wvls_n`f`, self.pixel_num)[cond]
   
    #     new_idx = self.m_n * self.wvls_n * self.proj_H * self.proj_W * grid_b_valid \
    #             + self.wvls_n * self.proj_H * self.proj_W * grid_m_valid \
    #             + self.proj_H * self.proj_W * grid_w_valid \
    #             + self.proj_W * r_proj_valid \
    #             + c_proj_valid

    #     return new_idx.long(), cond
            
    def grid_sample(self, uv1, hyp_f, illum_img):
        uv1_reshape = uv1.reshape(self.batch_size, self.m_n, self.wvls_n, 3, self.cam_H, self.cam_W)
        
        # split integer and decimal
        uv1_integer = uv1_reshape.long()
        u_float, v_float = uv1_reshape[:,:,:,0] - uv1_integer[:,:,:,0], uv1_reshape[:,:,:,1] - uv1_integer[:,:,:,1]

        weights = [
            (1-u_float) * (1-v_float),
            u_float * (1-v_float),
            (1-u_float) * v_float,
            u_float * v_float
        ]

        indices = ['A', 'B', 'C', 'D']
        final_illum_img = torch.zeros_like(illum_img)
        for idx, corner in enumerate(indices):
            new_idx, cond = self.get_newidx(uv1.long(), corner)
            valid_pattern_img = hyp_f[new_idx]
            illum_img[cond.flatten()] = valid_pattern_img.flatten()
            final_illum_img += weights[idx].flatten() * illum_img

        return final_illum_img.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)

    def get_newidx(self, uv1, corner):
        r_proj, c_proj = uv1[:,:,:,1].clone(), uv1[:,:,:,0].clone()

        if corner == 'B':
            r_proj += 1
        elif corner == 'C':
            c_proj += 1
        elif corner == 'D':
            r_proj += 1
            c_proj += 1

        cond = (0 <= r_proj) & (r_proj < self.proj_H) & (0 <= c_proj) & (c_proj < self.proj_W)
        r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]

        batch_samples = torch.linspace(0, self.batch_size-1, self.batch_size,device=self.device)
        wvl_samples = torch.linspace(0, self.wvls_n-1, self.wvls_n,device=self.device)
        m_samples = torch.linspace(0, self.m_n-1, self.m_n,device=self.device)

        pixel_samples = torch.linspace(0, self.pixel_num-1, self.pixel_num, device= self.device)
        grid_b, grid_m, grid_w, grid_pixel = torch.meshgrid(batch_samples,m_samples,wvl_samples,pixel_samples,indexing='ij')

        grid_b_valid = grid_b.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
        grid_m_valid = grid_m.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
        grid_w_valid = grid_w.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)[cond]
   
        new_idx = self.m_n * self.wvls_n * self.proj_H * self.proj_W * grid_b_valid \
                + self.wvls_n * self.proj_H * self.proj_W * grid_m_valid \
                + self.proj_H * self.proj_W * grid_w_valid \
                + self.proj_W * r_proj_valid \
                + c_proj_valid

        return new_idx.long(), cond
    
if __name__ == "__main__":
   
    import torch, os, sys
    sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
    from scipy.io import loadmat
    from hyper_sl.utils.ArgParser import Argument
    from hyper_sl.data import create_data_patch
    from hyper_sl.utils import data_process
    from scipy.interpolate import interp1d

    argument = Argument()
    arg = argument.parse()
    
    # 기존의 hyperpsectral 정보와 depth로 rendering 하는 코드
    create_data = create_data_patch.createData
    
    plane_XYZ = torch.tensor(loadmat('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/hyper_sl/image_formation/rendering_prac/plane_XYZ.mat')['XYZ_q'])
    plane_XYZ = data_process.crop(plane_XYZ)
    
    pixel_num = arg.cam_H * arg.cam_W
    random = False
    index = 0
    
    # depth = create_data(arg, "depth", pixel_num, random = random, i = index).create().unsqueeze(dim = 0).to(arg.device)
    # depth[:] = 0.8
    # depth[:] = plane_XYZ.reshape(-1,3)[:,2].unsqueeze(dim =0)*1e-3
    depth = torch.tensor(np.load("./calibration/dg_calibration_method2/20230728_data/spectralon_depth_0728_back.npy"), dtype=torch.float32).reshape(-1,3)[...,2].to(arg.device).unsqueeze(dim = 0)
    # depth = torch.tensor(np.load("./calibration/gray_code_depth/color_checker_depth_0508.npy"), dtype=torch.float32).reshape(-1,3)[...,2].unsqueeze(dim = 0)    
    
    normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0).to(arg.device)
    normal = torch.zeros_like(normal)
    normal[:,2] = -1
    
    hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0).to(arg.device)
    hyp = torch.ones_like(hyp)
    
    occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0).to(arg.device)
    occ = torch.ones_like(occ)
    
    cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0).to(arg.device)
    
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