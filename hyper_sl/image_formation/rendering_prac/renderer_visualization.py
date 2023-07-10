import numpy as np
import torch, os, sys
from tqdm import tqdm
from scipy.io import savemat

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation.projector import Projector
from hyper_sl.image_formation.camera import Camera
from hyper_sl.image_formation import distortion

from hyper_sl.data.create_data_patch import createData

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
        
        # visualize virtual plane for specific wavelength and m
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter(self.sensor_X_virtual[0,0].detach().cpu().flatten(), self.sensor_Y_virtual[0,0].detach().cpu().flatten())
        # ax.scatter(self.sensor_X_virtual[1,0].detach().cpu().flatten(), self.sensor_Y_virtual[1,0].detach().cpu().flatten())
        # ax.scatter(self.sensor_X_virtual[2,0].detach().cpu().flatten(), self.sensor_Y_virtual[2,0].detach().cpu().flatten())
        # plt.show()

        # save mat files
        # sensor_X_real = self.xyz_proj_dg[0]
        # sensor_Y_real = self.xyz_proj_dg[1]
        
        # lmb = list(np.linspace(400, 680, 29)*1e-9)
        # self.dat_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/dat_dispersion_coord"
        # for m in tqdm(range(len(self.m_list))):
        #     for i in tqdm(range(len(lmb)), leave = False):
        #         sensor_X_real = self.xyz_proj_dg[0]
        #         sensor_Y_real = self.xyz_proj_dg[1]
                
        #         savemat(os.path.join(self.dat_path, 'dispersion_coordinates_m%d_wvl%d.mat'%(self.m_list[m], lmb[i]*1e9 )), {'x': self.sensor_X_virtual[m,i].data.cpu().numpy(), 'y':self.sensor_Y_virtual[m,i].data.cpu().numpy(), 'xo': sensor_X_real.data.cpu().numpy(), 'yo': sensor_Y_real.data.cpu().numpy()})
        
        # check rays for a specific wavelength ==============================================================================================================================================

        m = torch.tensor([1,], device=self.device)
        # m_idx = 2
        
        lmb = list(torch.linspace(420, 660, 25)*1e-9)

        alpha_m_list = []
        beta_m_list = []
        z_list = []

        for i in range(len(lmb)):
            alpha_m = self.get_alpha_m(alpha_i=self.alpha_i, m =m, lmbda=lmb[i])
            beta_m = self.get_beta_m(beta_i=self.beta_i)
            z = self.get_z_m(alpha_m, beta_m)

            alpha_m_list.append(alpha_m)
            beta_m_list.append(beta_m)
            z_list.append(z)
        
        # lmb_ind = 0
        # colors_px = np.array([plt.cm.viridis(a) for a in np.linspace(0.0, 1.0, self.proj_H*self.proj_W)])
        
        # alpha_m = alpha_m_list[lmb_ind]
        # beta_m = beta_m_list[lmb_ind]
        # z = z_list[lmb_ind]
        # fig = plt.figure(figsize=(10,5))
        # ax = plt.axes(projection = '3d')
        # ax.set_xlim([-0.01,0.01])
        # ax.set_ylim([-0.01,0.01])
        # ax.set_zlim([-0.03,0.01])

        self.intersection_points_r, self.xyz_proj_dg = self.intersection_points_r.detach().cpu(), self.xyz_proj_dg.detach().cpu()
        alpha_m, beta_m, z = alpha_m.detach().cpu(), beta_m.detach().cpu(), z.detach().cpu()
        
        # for i in range(0,self.proj_H*self.proj_W,10000):
        #     start = [self.intersection_points_r[0,i],self.intersection_points_r[1,i],self.intersection_points_r[2,i]]

        #     # 역방향
        #     scale =  -1/10
        #     # diffracted ray
        #     X_d = [start[0], start[0] + scale*alpha_m[0,0,i]]
        #     Y_d = [start[1], start[1] + scale*beta_m[i]]
        #     Z_d = [start[2], start[2] + scale*z[0,0,i]]
        #     ax.plot(X_d,Y_d,Z_d, color = colors_px[i], linewidth = 1, linestyle = 'dashed')

        #     # projector sensor points
        #     ax.scatter(self.xyz_proj_dg[0,i], self.xyz_proj_dg[1,i], self.xyz_proj_dg[2,i], marker= 'o', color = 'red', s= 1)
        #     ax.scatter(self.sensor_X_virtual[m_idx,lmb_ind,i].detach().cpu(), self.sensor_Y_virtual[m_idx,lmb_ind,i].detach().cpu(), self.sensor_Z_virtual[m_idx,lmb_ind,i].detach().cpu(),  marker= 'o', color = 'blue', s= 1)

        # ax.scatter(self.proj_center_dg[0].detach().cpu(),self.proj_center_dg[1].detach().cpu(), self.proj_center_dg[2].detach().cpu(),  marker= 'o', color = 'green', s= 2)
        # ax.scatter(self.optical_center_virtual[m_idx,lmb_ind,0].detach().cpu(),self.optical_center_virtual[m_idx,lmb_ind,1].detach().cpu(), self.optical_center_virtual[m_idx,lmb_ind,2].detach().cpu(),  marker= 'o', color = 'black', s= 2)

        # plt.title('%d nm'%int(lmb[lmb_ind]*1e9))

        # ax.set_xlabel('x-axis')
        # ax.set_ylabel('y-axis')
        # ax.set_zlabel('z-axis')

        # ax.view_init(0, 90)
        # plt.show()
        # ==========================================================================================================================
        
        #  intersected points 에서 정방향의 diffracted dir그리기 for all wavelengths 
        fig = plt.figure(figsize=(10,5))
        ax = plt.axes(projection = '3d')
        ax.set_xlim([-0.01,0.01])
        ax.set_ylim([-0.05,0.005])
        ax.set_zlim([-0.05,0.03])

        colors = np.array([plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(lmb))])

        m_idx = 2
        # idx_list = [64300, 128300, 192300]
        # idx_list = [32100, 64100, 96100]
        
        for j in range(len(lmb)):
            j = -1
            alpha_m = alpha_m_list[j]
            beta_m = beta_m_list[j]
            z = z_list[j]

            for i in range(0,self.proj_H*self.proj_W,5000):
            # for i in idx_list:  
                start = [self.intersection_points_r[0,i],self.intersection_points_r[1,i],self.intersection_points_r[2,i]]

                # intersected points
                ax.scatter(start[0],start[1],start[2], marker = 'o', color = 'green', s = 1)

                # 정방향
                scale =  1/10
                # diffracted ray
                X_d = [start[0], start[0] + scale*alpha_m[0,0,i].detach().cpu()]
                Y_d = [start[1], start[1] + scale*beta_m[i].detach().cpu()]
                Z_d = [start[2], start[2] + scale*z[0,0,i].detach().cpu()]
                ax.plot(X_d,Y_d,Z_d, color = colors[j], linewidth = 0.25)
                ######
                
                
                #ax.plot(start[0], start[1], start[2], start[0] + 1/40*alpha_m[i,j], start[1] + 1/40*beta_m[i,j], start[2] +1/40*z[i,j])

                # virtual projector plane & center
                # ax.scatter(self.sensor_X_virtual[m_idx,j,i].detach().cpu(), self.sensor_Y_virtual[m_idx,j,i].detach().cpu(), self.sensor_Z_virtual[m_idx,j,i].detach().cpu(), s= 1, color = colors[j])                
                # ax.scatter(self.optical_center_virtual[m_idx,j,0].detach().cpu(), self.optical_center_virtual[m_idx,j,1].detach().cpu(), self.optical_center_virtual[m_idx,j,2].detach().cpu(), s=1 , color = colors[j])
                
                # 역방향
                scale =  -1/10
                # diffracted ray
                X_d = [start[0], start[0] + scale*alpha_m[0,0,i].detach().cpu()]
                Y_d = [start[1], start[1] + scale*beta_m[i].detach().cpu()]
                Z_d = [start[2], start[2] + scale*z[0,0,i].detach().cpu()]
                ax.plot(X_d,Y_d,Z_d, color = colors[j], linewidth = 0.1, linestyle = 'dashed')

                # projector sensor points
                ax.scatter(self.sensor_X_virtual[1,j,i].detach().cpu(), self.sensor_Y_virtual[1,j,i].detach().cpu(), self.sensor_Z_virtual[1,j,i].detach().cpu(), marker= 'o', color = 'blue', s= 1)

                start_proj = [self.optical_center_virtual[1,j,0].detach().cpu(), self.optical_center_virtual[1, j ,1].detach().cpu(), self.optical_center_virtual[1,j,2].detach().cpu()]
                # incident ray
                X_i = [start_proj[0], start[0]]
                Y_i = [start_proj[1], start[1]]
                Z_i = [start_proj[2], start[2]]
                ax.plot(X_i,Y_i,Z_i, color = 'black', linewidth = 0.25)

                ax.scatter(self.optical_center_virtual[1,j,0].detach().cpu(), self.optical_center_virtual[1,j,1].detach().cpu(), self.optical_center_virtual[1,j,2].detach().cpu(), marker= '*', c = 'cyan', s = 2)

                ## 1, -1 order scatter
                
                
                
                # x label
                # plt.xticks(rotation=30, fontsize = 10)
                # ax.xaxis.set_label_coords(0.5,-0.15)
                # ax.set_xlabel('x-axis', labelpad=20)
                ax.set_xticklabels([])
                plt.xlabel('')
                
                # y label
                ax.set_yticklabels([])
                plt.ylabel('')
                # z label
                # ax.zaxis.set_label_coords(0.5,-0.15)
                # ax.set_zlabel('z-axis', labelpad=10)
                ax.set_zlabel('')
                ax.set_zticks([])

        ax.view_init(0,90)
        # plt.savefig("image_formation_dispersion.svg")
        plt.show()
        
        # ==========================================================================================================================
        
        
        # extrinsic matrix of virtual proj(real proj to vir proj)
        self.extrinsic_proj_virtual = torch.zeros((self.m_n, self.wvls_n, 4, 4), device= self.device) 
        self.rot = torch.tensor([1,1,1])
        self.extrinsic_proj_virtual[:,:,0,0] = self.rot[0]
        self.extrinsic_proj_virtual[:,:,1,1] = self.rot[1]
        self.extrinsic_proj_virtual[:,:,2,2] = self.rot[2]
        
        self.optical_center_proj = torch.squeeze(self.optical_center_proj)

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
        
    def render(self, depth, normal, hyp, occ, cam_coord, eval):
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
        X,Y,Z = self.cam.unprojection(depth = depth, cam_coord = cam_coord)
        
        # shading term
        # B, m, 29, 3, # pixel
        illum_vec_unit = self.illum_unit(X,Y,Z, self.optical_center_virtual) 
        shading = (illum_vec_unit*normal_vec_unit_clip[:,None,:,:].unsqueeze(dim = 1)).sum(axis = 3)
        # shading = shading.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num) 
        
        # shading 반대로 된건 아닌지? 확인 필요함
        shading = shading.reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num)

        # project world coord onto vir proj(in Virtual proj coord)
        xy_vproj = self.proj.projection(X,Y,Z, self.extrinsic_proj_virtual, self.focal_length_proj_virtual)
                                
        # xy_vproj sensor coord in dg coord (to put in distortion function)
        xy_dg = self.proj.vproj_to_dg(xy_vproj, self.extrinsic_proj_virtual, self.sensor_Z_virtual, self.extrinsic_diff)
        xy_dg = xy_dg.to(self.device)  # TODO: do we need this? 
        
        # distortion coefficent 불러오기
        sensor_X_virtual_distorted, sensor_Y_virtual_distorted = self.dist.distort_func(xy_dg[..., 0,:], xy_dg[...,1,:], self.p_list[...,0,:], self.p_list[...,1,:])
        # sensor_X_virtual_distorted, sensor_Y_virtual_distorted = -sensor_X_virtual_distorted, -sensor_Y_virtual_distorted

        xy_proj = torch.stack((sensor_X_virtual_distorted, sensor_Y_virtual_distorted), dim=3)
        # change it to real proj coord
        xy_proj_real = self.proj.dg_to_rproj(xy_proj, self.extrinsic_diff)
        
        # normalization
        xy_proj_real_norm = self.normalize.normalization(self.arg, xy_proj_real[:,1,0,...].permute(0,2,1), train = True, xyz = False, proj = True)

        r_proj = xy_proj_real[:,:,:,0,:]/self.proj_pitch + self.proj_W/2
        c_proj = xy_proj_real[:,:,:,1,:]/self.proj_pitch + self.proj_H/2
        rc_proj = torch.cat((r_proj.unsqueeze(dim = 3), c_proj.unsqueeze(dim =3)), dim = 3)
        rc_proj = rc_proj.transpose(4,3).reshape(self.batch_size, self.m_n, self.wvls_n, self.pixel_num, 2) 
        rc_proj = rc_proj.to(self.device)  # TODO: do we need this?   
        
        r_proj, c_proj = rc_proj[...,1], rc_proj[...,0]

        cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W)
        r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]
        r_proj_valid, c_proj_valid = torch.tensor(r_proj_valid), torch.tensor(c_proj_valid)  # TODO: do we need this? 

        new_idx = self.proj_W * r_proj_valid.long() + c_proj_valid.long()  # TODO: use a variable instead of hard coding? 
        
        cam_N_img = torch.zeros(size=(self.batch_size, self.pixel_num, self.n_illum, 3), device= self.device)
        
        for j in range(self.n_illum):

            illum = self.load_data.load_illum(j).to(self.device)  # TODO: load this at the initialization and define it as the member variable for optimization
            
            illum_img = torch.zeros(self.batch_size, self.m_n, self.wvls_n, self.pixel_num, device= self.device).flatten()

            # max 2.8, min 0
            illum_hyp = illum.reshape((self.proj_H* self.proj_W, 3))@self.CRF_proj.T
            illum_hyp = illum_hyp.reshape((self.proj_H,self.proj_W,self.wvls_n))
            illum_hyp_unsq = torch.stack((illum_hyp,illum_hyp,illum_hyp), dim = 0)
            illum_hyp_unsq = illum_hyp_unsq.permute(0,3,1,2) # 3, 29, 720, 1440 
                            
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
            
        xy_proj_real_data = xy_proj_real[:,1,0,...].permute(0,2,1)

        render_end = time.time()
        
        print(f"render time : {render_end - render_start:.5f} sec")
        print(f'rendering finished for iteration')
        
        return cam_N_img, xy_proj_real_norm, xy_proj_real_data, illum_data, shading
    
    def proj_sensor_plane(self):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """
         # proj sensor
        xs = torch.linspace(0,self.proj_H-1, self.proj_H)
        ys = torch.linspace(0,self.proj_W-1, self.proj_W)
        r, c = torch.meshgrid(xs, ys, indexing='ij')
        
        c, r = c.flatten(), r.flatten()
        ones = torch.ones_like(c)
        cr1 = torch.stack((c,r,ones), dim = 0)
        xyz = (torch.linalg.inv(self.intrinsic_proj_real)@cr1)*self.proj_focal_length

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
    
    from hyper_sl.utils.ArgParser import Argument
    from hyper_sl.data import create_data_patch
    argument = Argument()
    arg = argument.parse()
    
    # 기존의 hyperpsectral 정보와 depth로 rendering 하는 코드
    create_data = create_data_patch.createData
    
    pixel_num = arg.cam_H * arg.cam_W
    random = False
    index = 0
    
    depth = create_data(arg, "depth", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
    cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0)
    
    # n_scene, random, pixel_num, eval
    cam_N_img, xy_proj_real_norm, xy_proj_real_data, illum_data, shading = PixelRenderer(arg).render(depth = depth, normal = normal, hyp = hyp, cam_coord = cam_coord, occ = occ, eval = True)
    
    print('end')