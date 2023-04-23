import numpy as np
import torch
import sys, os, cv2

sys.path.append('/workspace/Scalable-Hyperspectral-3D-Imaging')
from hyper_sl.utils import ArgParser
from hyper_sl.utils import load_data
from hyper_sl.image_formation import projector
from hyper_sl.image_formation import camera
from hyper_sl.utils import normalize

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
        
        # gt_xy_unnorm = self.normalize.un_normalization(gt_xy.unsqueeze(dim = 0))
        
        # ones = torch.ones_like(pred_xy_unnorm[0,...,0])
        # ones[:] = self.proj.focal_length_proj
        # pred_xyz_unnorm = torch.stack((pred_xy_unnorm[0,:,0], pred_xy_unnorm[0,:,1], ones))
        # suv = self.proj.intrinsic_proj_real()@pred_xyz_unnorm.detach().cpu()
        # uv1 = suv / suv[2]
        # # uv1 = uv1.reshape(3, 580*890)
        # uv1 = uv1.permute(1,0)[...,:2].unsqueeze(dim = 1)
        
        # # # Triangulation
        # camproj_calib_path = '/home/shshin/Scalable-Hyperspectral-3D-Imaging/calibration/calibration_propcam.xml'
        # fs = cv2.FileStorage(camproj_calib_path, cv2.FileStorage_READ)
        # img_shape = fs.getNode("img_shape").mat()
        # cam_int = fs.getNode("cam_int").mat()
        # cam_dist = fs.getNode("cam_dist").mat()
        # proj_int = fs.getNode("proj_int").mat()
        # proj_dist = fs.getNode("proj_dist").mat()
        # cam_proj_rmat = fs.getNode("rotation").mat()
        # cam_proj_tvec = fs.getNode("translation").mat()
        # F = fs.getNode("fundamental").mat()
        # E = fs.getNode("epipolar").mat()
        
        # # camera plane xyz 
        # xyz_cam, cr1_r = self.camera_plane_coord(cam_coord, eval)
        # cam_pts = cr1_r.reshape(3, 580* 890).permute(1,0)[...,:2].unsqueeze(dim = 1)
        # cam_pts = cam_pts.detach().cpu().numpy()
        # # triangulate
        # P0 = np.dot(cam_int, np.array([[1,0,0,0],
        #                             [0,1,0,0],
        #                             [0,0,1,0]]))
        # P1 = np.concatenate((np.dot(proj_int, cam_proj_rmat), np.dot(proj_int,cam_proj_tvec)), axis = 1)
        # triang_res = cv2.triangulatePoints(P0, P1, cam_pts, uv1.detach().cpu().numpy())
        # # cam coord point 3d
        # points_3d = cv2.convertPointsFromHomogeneous(triang_res.T).squeeze()
        
        # # xyz
        # R, C = 580, 890
        # cam_pts = cam_pts.astype(np.int16)
        # xyz = np.zeros((R,C,3))
        # xyz[cam_pts[:,0,1], cam_pts[:,0,0], 0]=points_3d[:,0]
        # xyz[cam_pts[:,0,1], cam_pts[:,0,0], 1]=points_3d[:,1]
        # xyz[cam_pts[:,0,1], cam_pts[:,0,0], 2]=points_3d[:,2]

        
        # predicted xy proj to world coord
        # B, xyz, #pixel/ 4,1 / 4,1
        xyz_proj_world, center_proj, center_world = self.xy_proj_world(pred_xy_unnorm)
        
        # camera plane xyz 
        xyz_cam, cr1_r  = self.camera_plane_coord(cam_coord, eval)
        
        # unit dir of cam & proj
        cam_dir = self.dir_cam(xyz_cam= xyz_cam)
        proj_dir = self.dir_proj(center_proj = center_proj, xyz_proj_world= xyz_proj_world)
        
        point3d_list = self.point3d(center_world, center_proj, cam_dir, proj_dir)
        point3d_list = point3d_list.reshape(-1, self.pixel_num, 3)
        
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
            cam_coord = cam_coord.reshape(-1, self.pixel_num, self.arg.patch_pixel_num, 3)[...,4,:]

        cr1_r = cam_coord.reshape(-1, self.pixel_num, 3).permute(0,2,1)
        
        xyz_cam = torch.linalg.inv(self.intrinsic_cam)@(cr1_r*self.focal_length_cam)

        return xyz_cam, cr1_r
    
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
    
    def depth_error(self, point3d_list, eval):
        if eval == True:
            # load data
            data_num = 1
            occ = self.load_data.load_occ(data_num)
            real_depth = self.load_data.load_depth(data_num)
        
            pred_depth = point3d_list[...,2].reshape(self.cam_H, self.cam_W)
            depth_error = abs(real_depth*occ - pred_depth*occ)
            
        else:
            pred_depth = point3d_list[...,2].reshape(self.cam_H, self.cam_W)
            depth_error = abs(real_depth*occ - pred_depth*occ)
            
        return depth_error
    
if __name__ == "__main__":

    argument = ArgParser.Argument()
    arg = argument.parse()
    
    from hyper_sl.data import create_data
    
    pixel_num = arg.cam_H * arg.cam_W
    
    x_proj = torch.tensor(np.load('/workspace/Scalable-Hyperspectral-3D-Imaging/prediction/prediction_xy_1470.npy'))
    gt_proj = torch.tensor(np.load('/workspace/Scalable-Hyperspectral-3D-Imaging/prediction/ground_truth_xy_1470.npy'))
    unnorm_gt = torch.tensor(np.load('/workspace/Scalable-Hyperspectral-3D-Imaging/prediction/ground_truth_xy_real_150.npy'))
    cam_coord = create_data.createData(arg, 'coord', pixel_num, random = False).create().unsqueeze(dim = 0)

    point3d_list, depth_error = depthReconstruction(arg).depth_reconstruction(x_proj, gt_proj, cam_coord, True)
    
    print('end')