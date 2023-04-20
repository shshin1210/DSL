import numpy as np
import torch
import sys, os

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
from hyper_sl.utils import ArgParser
from hyper_sl.utils import load_data
from hyper_sl.image_formation import projector
from hyper_sl.image_formation import camera
from hyper_sl.utils import normalize
from hyper_sl.utils import data_process

class depthReconstruction():
    def __init__(self, arg):
        # arg
        self.arg = arg
        self.device = arg.device
        
        # class
        self.load_data = load_data.load_data(arg)
        self.proj = projector.Projector(arg, device= self.device)
        self.cam = camera.Camera(arg)
        self.normalize = normalize
        self.dilation = data_process.dilation
        
        # intrinsic of proj & cam
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.focal_length_cam = arg.focal_length*1e-3
        self.intrinsic_cam = self.cam.intrinsic_cam()
        
        # proj
        self.focal_length_proj = arg.focal_length_proj *1e-3
        self.z_proj_world =  arg.focal_length_proj * 1e-3
        self.proj_H, self.proj_W = arg.proj_H, arg.proj_W
        self.extrinsic_proj_real = self.proj.extrinsic_proj_real()
    
    def depth_reconstruction(self, pred_xy, cam_coord, eval):
        # pixel num
        if eval == False:
            self.pixel_num = self.arg.num_train_px_per_iter // self.arg.patch_pixel_num
        else:
            self.pixel_num = self.cam_H * self.cam_W
        # batch size
        batch_size = pred_xy.shape[0] / self.pixel_num
        
        # reshape
        pred_xy = pred_xy.reshape(-1, self.pixel_num, 2) # B, # pixel num, 2
         
        # unnormalized gt proj xy
        pred_xy_unnorm = self.normalize.un_normalization(pred_xy)
        
        # predicted xy proj to world coord
        # B, xyz, #pixel/ 4,1 / 4,1
        xyz_proj_world, center_proj, center_world = self.xy_proj_world(pred_xy_unnorm)
        
        # camera plane xyz 
        xyz_cam = self.camera_plane_coord(cam_coord, eval)
        
        # unit dir of cam & proj
        cam_dir = self.dir_cam(xyz_cam= xyz_cam)
        proj_dir = self.dir_proj(center_proj = center_proj, xyz_proj_world= xyz_proj_world)
        
        point3d_list = self.point3d(center_world, center_proj, cam_dir, proj_dir)
        point3d_list = point3d_list.reshape(-1, self.pixel_num, 3)
        
        error = self.depth_error(point3d_list, 0)
        
        return point3d_list

    def xy_proj_world(self, xy_proj_unnorm):
        
        center_world = torch.tensor([0,0,0,1]).float()
        center_world = center_world.unsqueeze(dim =1)

        center_proj = self.extrinsic_proj_real@center_world.to(self.device)
        
        ones = torch.ones_like(xy_proj_unnorm[...,0]).unsqueeze(dim = 2)
        z = torch.zeros_like(xy_proj_unnorm[...,0]).unsqueeze(dim = 2)
        z[:] = self.focal_length_proj
        
        xyz1_proj = torch.concat((xy_proj_unnorm, z, ones), dim = 2)

        xyz1_proj_world = (self.extrinsic_proj_real)@(xyz1_proj.to(self.device).permute(0,2,1))
        xyz_proj_world = xyz1_proj_world[:,:3] # B , xyz, # pixel
        
        return xyz_proj_world, center_proj, center_world
        
    def camera_plane_coord(self, cam_coord, eval):
        if eval == False:
            cam_coord = cam_coord.reshape(-1, self.pixel_num, self.arg.patch_pixel_num, 3)[...,self.arg.patch_pixel_num // 2,:]

        cr1_r = cam_coord.reshape(-1, self.pixel_num, 3).permute(0,2,1)
        
        xyz_cam = torch.linalg.inv(self.intrinsic_cam)@(cr1_r*self.focal_length_cam)

        return xyz_cam
    
    def dir_proj(self, center_proj, xyz_proj_world):
        center_proj_world = center_proj[:3]
        proj_dir = - center_proj_world.unsqueeze(dim=0) + xyz_proj_world
        
        proj_dir = torch.tensor(proj_dir)
        proj_norm = torch.norm(proj_dir, dim = 1)

        proj_dir = proj_dir / proj_norm.unsqueeze(dim = 1)
        proj_dir = proj_dir.permute(0,2,1)

        return proj_dir
        
    def dir_cam(self, xyz_cam):
        
        cam_dir = torch.tensor(xyz_cam)
        cam_norm = torch.norm(cam_dir, dim = 1)

        cam_dir = cam_dir / cam_norm.unsqueeze(dim = 1)
        cam_dir = cam_dir.permute(0,2,1)
        
        return cam_dir        
        
    def intersect(self, P,dir):
        """P and dir are NxD arrays defining N lines.
        D is the dimension of the space. This function 
        returns the least squares intersection of the N
        lines from the system given by eq. 13 in 
        http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
        """
        projs = torch.eye(dir.shape[2], device=self.device) - torch.unsqueeze(dir, dim =3)* torch.unsqueeze(dir, dim =2)
        R = projs.sum(axis=1) 
        q = (projs @ torch.unsqueeze(P, dim = 3)).sum(axis=1)
        
        # solve the least squares problem for the 
        # intersection point p: Rp = q
        p = torch.linalg.lstsq(R,q,rcond=None)[0]

        return p

    def point3d(self, center_world, center_proj, cam_dir, proj_dir):
        
        cam_dir = cam_dir.reshape(-1,3).to(self.device)
        proj_dir = proj_dir.reshape(-1,3)
        
        center_cam_world = center_world[:3].to(self.device)
        center_proj_world = center_proj[:3]
    
        p = torch.vstack((center_cam_world.squeeze(dim=1), center_proj_world.squeeze(dim =1))).unsqueeze(dim=0)
        d = torch.stack((cam_dir, proj_dir), dim = 1)
        
        point = self.intersect(p, d).squeeze()

        
        return point
    
    def depth_error(self, point3d_list, i):
        # load data
        data_num = i
        
        occ = self.load_data.load_occ(data_num)
        dilated_occ = data_process.dilation(occ)
        
        real_depth = self.load_data.load_depth(data_num)
    
        pred_depth = point3d_list[...,2].reshape(self.cam_H, self.cam_W)
        depth_error = abs(real_depth*dilated_occ - pred_depth.detach().cpu().numpy()*dilated_occ)

        return depth_error
    
if __name__ == "__main__":

    argument = ArgParser.Argument()
    arg = argument.parse()
    
    from hyper_sl.data import create_data
    
    pixel_num = arg.cam_H * arg.cam_W
    
    x_proj = torch.tensor(np.load('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/prediction/prediction_xy_1020.npy'))
    gt_proj = torch.tensor(np.load('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/prediction/ground_truth_xy_1020.npy'))
    # unnorm_gt = torch.tensor(np.load('/workspace/Scalable-Hyperspectral-3D-Imaging/prediction/ground_truth_xy_real_150.npy'))
    cam_coord = create_data.createData(arg, 'coord', pixel_num, random = False).create().unsqueeze(dim = 0)

    point3d_list = depthReconstruction(arg).depth_reconstruction(x_proj, gt_proj, cam_coord, True)
    
    print('end')