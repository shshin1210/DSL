import numpy as np
import torch, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')

from hyper_sl.utils import noise,normalize,load_data

from hyper_sl.image_formation.etc.projector import Projector
from hyper_sl.image_formation.etc.camera import Camera
from hyper_sl.image_formation import distortion

from hyper_sl.data.create_data import createData

class PixelRenderer():
    """ Render for a single scene 
        which has specific # of pixels for N different patterns
        
    """
    def __init__(self, arg):
        
        self.arg = arg

        # device
        self.device = arg.device
        
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
        self.cam_resolution = arg.cam_H
        self.cam_sensor_width = arg.sensor_width*1e-3
        self.cam_focal_length = arg.focal_length*1e-3
        self.crf_cam_path = arg.camera_response
        self.CRF_cam = torch.tensor(np.load(os.path.join(self.crf_cam_path, 'CRF_cam.npy'))).to(self.device)
        self.cam_pitch = self.cam_sensor_width/ self.cam_resolution   

        # proj
        self.proj_sensor_diag = arg.sensor_diag_proj *1e-3
        self.proj_focal_length = arg.focal_length_proj *1e-3
        self.proj_H = arg.proj_H
        self.proj_W = arg.proj_W
        self.crf_proj_path = arg.projector_response
        self.CRF_proj = torch.tensor(np.load(os.path.join(self.crf_proj_path, 'CRF_proj.npy'))).to(self.device)
        self.sensor_width_proj = torch.sin(torch.atan2(torch.tensor(self.proj_H),torch.tensor(self.proj_H)))*self.proj_sensor_diag
        self.proj_pitch = self.sensor_width_proj/ self.proj_H

        # arguments
        self.wvls = arg.wvls
        self.n_illum = arg.illum_num
        self.m_num = arg.m_num
        self.wvl_num = arg.wvl_num
        
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
        self.incident_dir = self.proj_center_dg - self.xyz_proj_dg

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

        # virtual projector pixels 
        self.sensor_z = self.xyz_proj_dg[2,:].mean() 
        self.start = [self.intersection_points_r[0,:],self.intersection_points_r[1,:],self.intersection_points_r[2,:]]
        
        # find virtual proj sensor xyz in dg coordinate
        self.scale_sensor = (self.sensor_z-self.start[2])/self.z.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W)

        self.sensor_X_virtual = self.start[0] + self.scale_sensor*self.alpha_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
        self.sensor_Y_virtual = self.start[1] + self.scale_sensor*self.beta_m.reshape(self.m_n, self.wvls_n, self.proj_H*self.proj_W) 
        self.sensor_Z_virtual = self.start[2] + self.scale_sensor*self.z.reshape(self.m_n, self.wvls_n, self.proj_H* self.proj_W)
        self.sensor_Z_virtual = self.sensor_Z_virtual.to(self.device)
        
        self.p = self.intersection_points_r.T
        self.d = self.diffracted_dir_unit_r.T
        
        # finding point p
        self.optical_center_virtual = self.proj.get_virtual_center(self.p,self.d, self.wvls_n, self.m_n)
        
        # optical_center_virtual shape : m_N, wvls_N, 3
        self.optical_center_virtual = torch.tensor(self.optical_center_virtual, dtype=torch.float32, device= self.device) 
        self.optical_center_proj = self.proj_center_dg.to(self.device) 
        
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
        self.focal_length_proj_virtual = (self.optical_center_virtual[:,:,2] - self.sensor_Z_virtual.mean()).abs() 
        self.focal_length_proj_virtual = self.focal_length_proj_virtual.to(self.device)
        self.cx_proj_virtual = self.sensor_X_virtual.mean() - self.optical_center_virtual[:,:,0] 
        self.cy_proj_virtual = self.sensor_Y_virtual.mean() - self.optical_center_virtual[:,:,1] 

        # distortion coefficient
        self.p_list = self.dist.bring_distortion_coeff(arg, self.m_list, self.wvls, self.dat_dir)
        self.p_list = self.p_list.to(device=self.device)

    # def render(self, N_scene, N_illum):
    def render(self, n_scene, random , pixel_num , eval):
        ### init 함수가 이미 불러졌으면 부르지 말기
        ### init 함수가 불러지지 않은 상태라면 부르기
        # if self.__init__(self.arg, n_scene, random, pixel_num, self.eval) == True:
        #     print('true')
        #     pass
        # else:
        #     print('false')
        
        print('rendering start')
        
        # N_illum, depth, normal, hyp, occ, pixel_num
        
        if eval == False:
            
            scene_data = torch.zeros(size=(n_scene, pixel_num, self.n_illum, 3))
            xy_proj_data = torch.zeros(size=(n_scene, pixel_num, 2))
            xy_proj_real_data = torch.zeros(size=(n_scene, pixel_num, 2))
            hyp_gt = torch.zeros(size =(n_scene, pixel_num, self.wvls_n))
            
        for i in range(n_scene):
            print(f'{i}-th scene rendering...')
            depth = self.create_data(self.arg, "depth", pixel_num, random = random, i = i).create()
            normal = self.create_data(self.arg, "normal", pixel_num, random = random, i = i).create()
            hyp = self.create_data(self.arg, 'hyp', pixel_num, random = random, i = i).create()
            occ = self.create_data(self.arg, 'occ', pixel_num, random = random, i = i).create()
    
            occ = occ.to(self.device)
            hyp = torch.tensor(hyp, device= self.device)
            
            normal_norm = torch.norm(normal, dim = 0)
            normal_vec_unit = normal/normal_norm
            normal_vec_unit = normal_vec_unit.to(self.device)
            normal_vec_unit_clip = torch.clamp(normal_vec_unit, 0, 1)
            
            #### Scene dependent variables
            # TODO: remove the per-data forloop and replace it with batch
            # for i in range(N_scene):  

            # depth2XYZ
            X,Y,Z = self.cam.unprojection(depth=depth)
            
            # shading term
            illum_vec_unit = self.illum_unit(X,Y,Z, self.optical_center_virtual) 
            shading = (illum_vec_unit*normal_vec_unit_clip).sum(axis = 2)
            shading = shading.reshape(self.m_n, self.wvls_n, self.cam_resolution, self.cam_resolution)
            
            # project world coord onto vir proj(in Virtual proj coord)
            xy_vproj = self.proj.projection(X,Y,Z, self.extrinsic_proj_virtual, self.focal_length_proj_virtual)
                                    
            # xy_vproj sensor coord in dg coord (to put in distortion function)
            xy_dg = self.proj.vproj_to_dg(xy_vproj, self.extrinsic_proj_virtual, self.sensor_Z_virtual)
            xy_dg = xy_dg.to(self.device)  # TODO: do we need this? 
            
            # distortion coefficent 불러오기
            sensor_X_virtual_distorted, sensor_Y_virtual_distorted = self.dist.distort_func(xy_dg[:, :, 0,:], xy_dg[:, :,1,:], self.p_list[:,:,0,:], self.p_list[:,:,1,:])
            sensor_X_virtual_distorted, sensor_Y_virtual_distorted = -sensor_X_virtual_distorted, -sensor_Y_virtual_distorted

            xy_proj = torch.stack((sensor_X_virtual_distorted, sensor_Y_virtual_distorted), dim=2)
            # change it to real proj coord
            xy_proj_real = self.proj.dg_to_rproj(xy_proj)
            
            # normalization
            xy_proj_real_norm = self.normalize.normalization(xy_proj_real[1,0,...].permute(1,0), train = False, xyz = False)
            # xy to rc (in real proj coord)
            # xy_proj_real = xy_proj_real.cpu().detach().numpy()  # TODO: shouldn't this be still on gpu?             

            r_proj = xy_proj_real[:,:,0,:]/self.proj_pitch + self.proj_W/2
            c_proj = xy_proj_real[:,:,1,:]/self.proj_pitch + self.proj_H/2
            rc_proj = torch.cat((r_proj.unsqueeze(dim = 2), c_proj.unsqueeze(dim =2)), dim = 2)
            rc_proj = rc_proj.transpose(3,2).reshape(self.m_n, self.wvls_n, self.cam_resolution, self.cam_resolution, 2) 
            rc_proj = rc_proj.to(self.device)  # TODO: do we need this?   
            
            r_proj, c_proj = rc_proj[:,:,:,:,1], rc_proj[:,:,:,:,0] # y = r , x = c

            cond = (0<= r_proj)*(r_proj < self.proj_H)*(0<=c_proj)*(c_proj< self.proj_W)
            r_proj_valid, c_proj_valid = r_proj[cond], c_proj[cond]
            r_proj_valid, c_proj_valid = torch.tensor(r_proj_valid), torch.tensor(c_proj_valid)  # TODO: do we need this? 

            new_idx = self.proj_W * r_proj_valid.long() + c_proj_valid.long()  # TODO: use a variable instead of hard coding? 
            
            cam_N_img = torch.zeros(size=(pixel_num, self.n_illum, 3), device= self.device)
            
            for j in range(self.n_illum):

                illum = self.load_data.load_illum(j).to(self.device)  # TODO: load this at the initialization and define it as the member variable for optimization
                
                illums_w_occ = torch.zeros((self.m_n, self.cam_resolution, self.cam_resolution, self.wvls_n))  # TODO: set device
                illum_img = torch.zeros(self.m_n, self.wvls_n, cond.shape[2], cond.shape[3], device=self.device).flatten()

                # max 2.8, min 0
                illum_hyp = illum.reshape((self.proj_H* self.proj_W, 3))@self.CRF_proj.T
                illum_hyp = illum_hyp.reshape((self.proj_H,self.proj_W,self.wvls_n))
                illum_hyp_unsq = torch.stack((illum_hyp,illum_hyp,illum_hyp), dim = 0)
                illum_hyp_unsq = illum_hyp_unsq.permute(0,3,1,2) 
                                
                hyp_f = illum_hyp_unsq.flatten() # mean : 1.0566

                valid_pattern_img = hyp_f[new_idx]
                
                illum_img[cond.flatten()] = valid_pattern_img
                
                illum_img = illum_img.reshape(self.m_n, self.wvls_n, self.cam_resolution, self.cam_resolution)
                # illums_m_img = illum_img.sum(axis = 0).reshape(self.wvls_n, self.cam_resolution * self.cam_resolution).permute(1,0) # no shading illum

                # multipy with occlusion
                illum_w_occ = illum_img*occ #
                                
                # final pattern
                illums_w_occ = illum_w_occ*shading # max 0.0093?
                # illums_w_occ = illum_w_occ
                illums_w_occ = illums_w_occ.permute(0,2,3,1)

                # illums_m_img = illums_w_occ.sum(axis = 0).reshape(self.cam_resolution * self.cam_resolution, self.wvls_n)

                # illum data                
                cam_m_img = torch.zeros((self.m_n, self.cam_resolution, self.cam_resolution, 3))  # TODO: set device
                
                # m order에 따른 cam img : cam_m_img
                for k in range(self.m_n):
                    cam_m_img[k] = (hyp * illums_w_occ[k] @ self.CRF_cam)

                cam_img = cam_m_img.sum(axis=0)
                # cam_img = cam_m_img[1]
                
                # rendering result, xy vproj return
                cam_N_img[:,j,:] = cam_img.reshape(self.cam_resolution*self.cam_resolution,3)
                
            # if eval == False:
            #     noise = self.noise.sample(cam_N_img.shape)
            #     cam_N_img += noise
            
            # hyp normalize
            hyp = hyp.reshape(self.cam_resolution*self.cam_resolution, self.wvls_n)
            
            scene_data[i, ...] = cam_N_img
            xy_proj_data[i,...] = xy_proj_real_norm
            xy_proj_real_data[i,...] = xy_proj_real[1,0,...].permute(1,0)
            hyp_gt[i, ...] = hyp
            
        return scene_data, xy_proj_data, xy_proj_real_data, hyp_gt
    
    def proj_sensor_plane(self):
        """ Projector sensor plane coordinates
        
            returns projector center coordinate, sensor plane coordiante
        
        """
        # proj sensor
        xs = torch.linspace(0,self.proj_H-1, self.proj_H)
        ys = torch.linspace(0,self.proj_W-1, self.proj_W)
        c, r = torch.meshgrid(xs, ys, indexing='ij')
        
        # projector's x,y,z coords to 실제 단위
        x_c, y_c = (r-self.proj_W/2)*self.proj_pitch, (c-self.proj_H/2)*self.proj_pitch

        z_c = torch.zeros_like(x_c)
        z_c[:] = -self.proj_focal_length

        # proj_center
        proj_center = torch.zeros(size=(4,1), device=self.device)
        proj_center[3,0] = 1

        # reshape to 720*720
        x_c,y_c,z_c = x_c.flatten(), y_c.flatten(), z_c.flatten()

        # make projector sensor xyz1 vector
        xyz1 = torch.stack((x_c,y_c,z_c, torch.ones_like(x_c)), dim = 1).transpose(0,1)
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
        lmbda = torch.unsqueeze(lmbda, dim = 0)
        alpha_i = alpha_i.unsqueeze(dim = 0)

        m_l_d = m*lmbda/d 
                    # 3, 77, 1
        alpha_m = - m_l_d.unsqueeze(dim = 2) + alpha_i # 3, 77, 518400

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
        XYZ = torch.stack((X,Y,Z), dim = 0).to(self.device)
        # optical_center_virtual : m_N, wvls_N, 3
        m_N = optical_center_virtual.shape[0]
        wvls_N = optical_center_virtual.shape[1]
        # optical center virtual in dg coord to world coord
        # optical_center_virtual = optical_center_virtual.squeeze(dim=0)
        ones = torch.ones(size = (optical_center_virtual[:,:,0].shape[0],optical_center_virtual[:,:,0].shape[1]), device= self.device)
        optical_center_virtual1 = torch.stack((optical_center_virtual[:,:,0],optical_center_virtual[:,:,1],optical_center_virtual[:,:,2], ones), dim = 2)
        optical_center_virtual1 = torch.unsqueeze(optical_center_virtual1, dim = 3)
        # 3, 77, 4, 1
                                                            #4,4                       # 4,1
        optical_center_virtual_world = self.extrinsic_proj_real@self.extrinsic_diff@optical_center_virtual1 # 4, m_N, wvls_N
        optical_center_virtual_world = optical_center_virtual_world[:,:,:3] # m_N, wvls_N, 3
        optical_center_virtual_world = torch.unsqueeze(optical_center_virtual_world, dim = 3)

        # XYZ = XYZ.permute(1,2,0) # 640, 640, 3
        # optical_center_virtual_world = optical_center_virtual_world.permute(1,2,0) # m_N, wvls_N, 3
        # illumination vector in world coord
        illum_vec = optical_center_virtual_world - XYZ 

        illum_norm = torch.norm(illum_vec, dim = 2) # dim = 0
        illum_norm = torch.unsqueeze(illum_norm, dim = 2)
        
        illum_unit = illum_vec/illum_norm
        illum_unit = illum_unit.reshape(m_N, wvls_N, 3, self.cam_resolution* self.cam_resolution) 

        return illum_unit
            
        
        
if __name__ == "__main__":
    from hyper_sl.utils.ArgParser import Argument
    
    argument = Argument()
    arg = argument.parse()
    
    # 기존의 hyperpsectral 정보와 depth로 rendering 하는 코드
    # img_hyp_text_dir = "C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset/image_formation/img_hyp_text"
    # img_hyp_file = "img_hyp_text_0000.npy"
    
    # depth = load_data.load_data(arg).load_depth(0).reshape(arg.cam_H*arg.cam_H)
    # hyp = np.load(os.path.join(img_hyp_text_dir, img_hyp_file)).reshape(arg.cam_H*arg.cam_H, 29)
    # hyp = torch.tensor(hyp, dtype = torch.float32)
    
    pixel_num = arg.cam_W*arg.cam_W
    # cam_coord = pixel_create_data.createData(arg,'coord', pixel_num, random=False).create()
    
    # n_scene, random, pixel_num, eval
    cam_N_img, xy_proj_norm, xyz_norm = PixelRenderer(arg, n_scene=1, random=False, pixel_num=640*640, eval = True).render()   
    
    cam_N_img = cam_N_img.detach().cpu().numpy()
    xy_proj_norm = xy_proj_norm.detach().cpu().numpy()
    # cam_coord_norm = cam_coord_norm.detach().cpu().numpy()
    xyz_norm = xyz_norm.detach().cpu().numpy()
    
    np.save('./cam_N_img.npy', cam_N_img)
    np.save('./xy_proj_norm.npy', xy_proj_norm)
    # np.save('./cam_coord_norm.npy', cam_coord_norm)
    np.save('./xyz_norm.npy', xyz_norm)
    
    print('end')