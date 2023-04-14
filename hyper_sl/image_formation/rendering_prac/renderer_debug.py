import numpy as np
import torch, os, sys
from tqdm import tqdm
from scipy.io import savemat
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation.projector import Projector
from hyper_sl.image_formation.camera import Camera
from hyper_sl.image_formation import distortion

from hyper_sl.data.create_data import createData

import time, math
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
        self.create_data = createData
        
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
        # self.result_dir = arg.render_result_dir
        
        # m order, wvls
        self.m_list = arg.m_list.to(device=self.device)
        self.m_n = len(self.m_list)
        self.wvls = arg.wvls.to(device=self.device)
        self.wvls_n = len(self.wvls)
        
        # extrinsic matrix
        self.extrinsic_proj_real = self.proj.get_ext_proj_real()
        self.extrinsic_diff = self.proj.get_ext_diff()
        
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
        
        # virtual projector pixels
        self.sensor_z = -0.01
        self.start = [self.intersection_points_r[0,:],self.intersection_points_r[1,:],self.intersection_points_r[2,:]]
        
        # find virtual proj sensor xyz in dg coordinate
        self.scale_sensor = (self.sensor_z-self.start[2])/self.z.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)

        self.sensor_X_virtual = self.start[0] + self.scale_sensor*self.alpha_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
        self.sensor_Y_virtual = self.start[1] + self.scale_sensor*self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
        self.sensor_Z_virtual = self.start[2] + self.scale_sensor*self.z.reshape(self.m_n, self.wvls_n, self.proj_H* self.proj_W)
        self.sensor_Z_virtual = self.sensor_Z_virtual.to(self.device)
        
        # real proj pixels
        # virtual projector pixels
        self.sensor_z = self.xyz_proj_dg[2,:].mean()
        self.start = [self.intersection_points_r[0,:],self.intersection_points_r[1,:],self.intersection_points_r[2,:]]
        
        self.scale_sensor = (self.sensor_z-self.start[2])/self.z.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)[1]

        self.sensor_X_virtual[1] = self.start[0] + self.scale_sensor*self.alpha_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)[1]
        self.sensor_Y_virtual[1] = self.start[1] + self.scale_sensor*self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)[1]
        self.sensor_Z_virtual[1] = self.start[2] + self.scale_sensor*self.z.reshape(self.m_n, self.wvls_n, self.proj_H* self.proj_W)[1]
        self.sensor_Z_virtual = self.sensor_Z_virtual.to(self.device)
                
        # extrinsic matrix of virtual proj(real proj to vir proj)
        self.extrinsic_proj_virtual = torch.zeros((self.m_n, self.wvls_n, 4, 4), device= self.device) 
        self.rot = torch.tensor([1,1,1])
        self.extrinsic_proj_virtual[:,:,0,0] = self.rot[0]
        self.extrinsic_proj_virtual[:,:,1,1] = self.rot[1]
        self.extrinsic_proj_virtual[:,:,2,2] = self.rot[2]
        
        self.optical_center_proj = torch.squeeze(self.optical_center_proj)
        
        # translation vector virtual plane to real proj plane
        self.trans = self.optical_center_virtual - self.optical_center_proj 

        self.extrinsic_proj_virtual[:,:,0,3] = self.trans[:,:,0]
        self.extrinsic_proj_virtual[:,:,1,3] = self.trans[:,:,1]
        self.extrinsic_proj_virtual[:,:,2,3] = self.trans[:,:,2]
        self.extrinsic_proj_virtual[:,:,3,3] = 1
        
        # intrinsic of virtual projector : intrinsic_proj_virtual
        self.focal_length_proj_virtual = (self.optical_center_virtual[:,:,2] - self.sensor_Z_virtual.mean(dim=2)).abs() 
        self.focal_length_proj_virtual = self.focal_length_proj_virtual.to(self.device)
        self.cx_proj_virtual = self.sensor_X_virtual.mean(dim=2) - self.optical_center_virtual[:,:,0] 
        self.cy_proj_virtual = self.sensor_Y_virtual.mean(dim=2) - self.optical_center_virtual[:,:,1] 

        # distortion coefficient
        self.p_list = self.dist.bring_distortion_coeff(arg, self.m_list, self.wvls, self.dat_dir)
        self.p_list = self.p_list.to(device=self.device)

    def render(self, depth, normal, hyp, occ, cam_coord, eval, illum_opt = None):
        print('rendering start')
        render_start = time.time()
        math.factorial(100000)
        
        self.batch_size = depth.shape[0]
        self.pixel_num = depth.shape[1]

        illum_data = torch.zeros(size =(self.batch_size, self.pixel_num, self.n_illum, self.wvls_n))
        
        occ = occ.to(self.device).unsqueeze(dim = 1)
        hyp = torch.tensor(hyp, device= self.device)
        
        normal_norm = torch.norm(normal.permute(1,0,2).reshape(3, -1), dim = 0)
        normal_norm = normal_norm.reshape(self.batch_size, self.pixel_num).unsqueeze(dim = 1)
        
        normal_vec_unit = normal/normal_norm
        normal_vec_unit = normal_vec_unit.to(self.device)
        normal_vec_unit_clip = torch.clamp(normal_vec_unit, 0, 1) # B, 3, #pixel
    
        # depth2XYZ
        X, Y, Z = self.cam.unprojection(depth = depth, cam_coord = cam_coord)
        
        # shading term
        # shading 반대로 된건 아닌지? 확인 필요함
        illum_vec_unit = self.illum_unit(X,Y,Z, self.optical_center_virtual) 
        shading = (illum_vec_unit*normal_vec_unit_clip[:,None,:,:].unsqueeze(dim = 4)).sum(axis = 3)
        shading = abs(shading)
        shading = shading.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num) 
        
        # project world coord onto vir proj(in Virtual proj coord)
        xyz_vproj = self.proj.projection(X,Y,Z, self.extrinsic_proj_virtual, self.focal_length_proj_virtual)
                
        # xy_vproj sensor coord in dg coord (to put in distortion function)
        xyz_dg = self.proj.vproj_to_dg(xyz_vproj, self.extrinsic_proj_virtual, self.extrinsic_diff)
        xyz_dg = xyz_dg.to(self.device)  # TODO: do we need this? 
        
        # distortion coefficent 불러오기
        sensor_X_virtual_distorted, sensor_Y_virtual_distorted = self.dist.distort_func(xyz_dg[..., 0,:]/self.proj_pitch, xyz_dg[...,1,:]/self.proj_pitch, self.p_list[...,0,:], self.p_list[...,1,:])
        
        xy_dg = torch.stack((sensor_X_virtual_distorted*self.proj_pitch, sensor_Y_virtual_distorted*self.proj_pitch), dim=3)
        
        # change it to real proj coord
        xyz_proj_real = self.proj.dg_to_rproj(xy_dg, self.extrinsic_diff)       
        
        # normalization
        xy_proj_real_norm = self.normalize.normalization(self.arg, xyz_proj_real[:,1,0,...].permute(0,2,1), train = True, xyz = False, proj = True)

        uv1_proj = (self.intrinsic_proj_real.to(self.device)@xyz_proj_real)/self.proj_focal_length
        r_proj, c_proj = uv1_proj[:,:,:,0], uv1_proj[:,:,:,1]

        rc_proj = torch.cat((r_proj.unsqueeze(dim = 3), c_proj.unsqueeze(dim =3)), dim = 3)
        rc_proj = rc_proj.transpose(4,3).reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num, 2) 
        rc_proj = rc_proj.to(self.device)  # TODO: do we need this?   
        
        r_proj, c_proj = rc_proj[...,1], rc_proj[...,0]

        cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W)
        r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]
        r_proj_valid, c_proj_valid = torch.tensor(r_proj_valid), torch.tensor(c_proj_valid)  # TODO: do we need this? 

        new_idx = self.proj_W * r_proj_valid.long() + c_proj_valid.long()  # TODO: use a variable instead of hard coding? 
        
        cam_N_img = torch.zeros(size=(self.batch_size, self.pixel_num, self.n_illum, 3), device= self.device)
        
        # for j in range(self.n_illum):
        for j in range(1):
            # illum 받은거
            if illum_opt == None:
                illum = self.load_data.load_illum(j).to(self.device)  # TODO: load this at the initialization and define it as the member variable for optimization
            else:
                illum = illum_opt[j]
            # illum = self.load_data.load_illum(j).to(self.device)  # TODO: load this at the initialization and define it as the member variable for optimization
            
            illum_img = torch.zeros(self.batch_size, self.m_n, self.wvls_n, self.pixel_num, device= self.device).flatten()

            # max 2.8, min 0
            illum_hyp = illum.reshape((self.proj_H*self.proj_W, 3))@self.CRF_proj.T
            illum_hyp = illum_hyp.reshape((self.proj_H,self.proj_W,self.wvls_n))
            illum_hyp_unsq = torch.stack((illum_hyp,illum_hyp,illum_hyp), dim = 0)
            illum_hyp_unsq = illum_hyp_unsq.permute(0,3,1,2)
                            
            hyp_f = illum_hyp_unsq.flatten() # mean : 1.0566

            valid_pattern_img = hyp_f[new_idx]
            
            illum_img[cond.flatten()] = valid_pattern_img
            
            illum_img = illum_img.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)
            illums_m_img = illum_img.sum(axis = 1).reshape(self.batch_size, self.wvls_n, self.pixel_num).permute(0,2,1)
            
            # multipy with occlusion
            illum_w_occ = illum_img*occ.unsqueeze(dim=1)
                            
            # final pattern
            illums_w_occ = illum_w_occ*shading # max 0.0093?
            illums_w_occ = illums_w_occ.permute(0,1,3,2) # illums_w_occ.permute(0,2,3,1)
        
            cam_m_img = torch.zeros((self.batch_size, self.m_n, self.pixel_num, 3))
            
            # m order에 따른 cam img : cam_m_img
            for k in range(self.m_n): 
                cam_m_img[:,k,...] = (hyp*illums_w_occ[:,k,...] @ self.CRF_cam)
            
            cam_img = cam_m_img.sum(axis=1)
            # cam_img = cam_m_img[:,1,...]
            
            # rendering result, xy vproj return
            cam_N_img[...,j,:] = cam_img
            illum_data[:,:,j,:] = illums_m_img

        if eval == False:
            noise = self.noise.sample(cam_N_img.shape)
            cam_N_img += noise
            
        xy_proj_real_data = xyz_proj_real[:,1,0,...].permute(0,2,1)

        render_end = time.time()
        
        print(f"render time : {render_end - render_start:.5f} sec")
        print(f'rendering finished for iteration')
        
        a = cam_N_img[0].reshape(768,1024,42,3).detach().cpu()
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(a[:,:,0]*100)
        plt.show()
        
        return cam_N_img, xy_proj_real_norm, xy_proj_real_data, illum_data, shading
    
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
    
    def illum_unit(self, X,Y,Z, optical_center_virtual):
        """ 
        inputs : X,Y,Z world coord
                virtual projector center in dg coord (optical_center_virtual) # need to change it to world coord
        """
        XYZ = torch.stack((X,Y,Z), dim = 1).to(self.device)
        XYZ = XYZ.unsqueeze(dim=1)
        
        # optical_center_virtual : m_N, wvls_N, 3(x,y,z)
        m_N = optical_center_virtual.shape[0]
        wvls_N = optical_center_virtual.shape[1]
        
        # optical center virtual in dg coord to world coord
        ones = torch.ones(size = (m_N, wvls_N), device= self.device)
        optical_center_virtual1 = torch.stack((optical_center_virtual[:,:,0],optical_center_virtual[:,:,1],optical_center_virtual[:,:,2], ones), dim = 2)
        optical_center_virtual1 = torch.unsqueeze(optical_center_virtual1, dim = 3)
        optical_center_virtual_world = self.extrinsic_proj_real@self.extrinsic_diff@optical_center_virtual1 # 4, m_N, wvls_N
        optical_center_virtual_world = optical_center_virtual_world[:,:,:3] # m_N, wvls_N, 3

        # illumination vector in world coord
        illum_vec = optical_center_virtual_world - XYZ.unsqueeze(dim = 1)

        illum_norm = torch.norm(illum_vec, dim = 3) # dim = 0
        illum_norm = torch.unsqueeze(illum_norm, dim = 3)
        
        illum_unit = illum_vec/illum_norm
        illum_unit = illum_unit.reshape(self.batch_size, m_N, wvls_N, 3, -1) 

        return illum_unit
            
        
        
if __name__ == "__main__":
    from hyper_sl.utils.ArgParser import Argument
    from scipy.io import loadmat
    from hyper_sl.utils.ArgParser import Argument
    from hyper_sl.data import create_data
    argument = Argument()
    arg = argument.parse()
    
    # 기존의 hyperpsectral 정보와 depth로 rendering 하는 코드
    create_data = create_data.createData
    
    pixel_num = 1
    random = False
    index = 0
    
    plane_XYZ = loadmat('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/hyper_sl/image_formation/plane_XYZ.mat')['XYZ_q']
    depth_cam = plane_XYZ[558, 259]*1e-3
    
    depth = torch.zeros(size=(pixel_num, )).unsqueeze(dim =0)
    depth[:] = depth_cam[2]
        
    normal = torch.ones(size=(pixel_num, 3)).unsqueeze(dim =0)
    hyp = torch.ones(size=(pixel_num, 29)).unsqueeze(dim =0)
    occ = torch.ones(size=(pixel_num, )).unsqueeze(dim =0)

    # cam_coord = torch.tensor([[223.7971,559.8687,1]]).unsqueeze(dim =0)
    # cam_coord = torch.tensor([[259.7466,560.8459,1]]).unsqueeze(dim =0)
    cam_coord = torch.tensor([[259.3319,558.5748,1]]).unsqueeze(dim =0)
    
    import cv2
    illum = cv2.imread("C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/illum/gird_360.png").astype(np.float32)
    illum = illum / 255.
    illum = torch.tensor(illum, device='cuda').unsqueeze(dim = 0)

    # n_scene, random, pixel_num, eval
    cam_N_img, xy_proj_real_norm, xy_proj_real_data, illum_data, shading = PixelRenderer(arg).render(depth = depth, normal = normal, hyp = hyp, cam_coord = cam_coord, occ = occ, eval = True, illum_opt=illum)
    
    print('end')