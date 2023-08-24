import numpy as np
import cv2, os,sys
import scipy.io as io
import torch

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
import matplotlib.pyplot as plt
from hyper_sl.utils.ArgParser import Argument
from point_process import PointProcess
from hyper_sl.utils import calibrated_params

class PointDistortion():
    def __init__(self, arg, date, position):
        # arguments
        self.arg = arg
        self.date = date
        self.wvls = np.array([430, 450, 480, 500, 520, 550, 580, 600, 620, 650, 660])
        self.position = position
        self.total_px = arg.total_px
        cam_int, _ = calibrated_params.bring_params(arg.calibration_param_path, "cam")
        self.cam_int = torch.tensor(cam_int, device=self.arg.device)
        
        # directory 
        self.main_dir = "./calibration/dg_calibration_method2/2023%s_data"%self.date
        self.data_dir = os.path.join(self.main_dir, self.position)
        self.processed_data_dir = os.path.join(self.main_dir, "%s_processed"%self.position)
        self.points_dir = self.data_dir + '_points' # detected points dir
        self.points_3d_dir = os.path.join(self.main_dir , "spectralon_depth_%s_%s.npy"%(self.date,self.position)) # spectralon 3d points
        self.pattern_npy_dir = "./calibration/dg_calibration_method2/grid_npy" # pattern npy 
        self.final_dir = './calibration/dg_calibration_method2/2023%s_data/%s_processed_homo'%(date, position)

        # point datas
        # self.filter_zero_orders, self.no_filter_zero_orders = self.load_points()
        # np.save('./filter_zero_orders_%s.npy'%position, self.filter_zero_orders)
        # np.save('./no_filter_zero_orders_%s.npy'%position, self.no_filter_zero_orders)
        
        self.filter_zero_orders = torch.tensor(np.load('./filter_zero_orders_%s.npy'%position), device= self.arg.device)
        self.no_filter_zero_orders = torch.tensor(np.load('./no_filter_zero_orders_%s.npy'%position), device= self.arg.device)
        
    def get_proj_px(self, dir):
        """
            get (u, v) coords for projected projector sensor plane
        """
        proj_px = np.load(dir)
        
        return proj_px
    
    def get_detected_pts(self, i, proj_px):
        """
            get preprocessed detection points : m, wvl, # points, 3(xyz)
            
            returns detected points
            
        """
        detected_pts_dir = self.points_dir + '/pattern_%04d'%i
        processed_img_dir = self.processed_data_dir + '/pattern_%04d'%i
        detected_pts = PointProcess(self.arg, self.data_dir, detected_pts_dir, processed_img_dir, self.wvls, i, proj_px, self.position).point_process()
        # detected_pts = (np.round(detected_pts)).astype(np.int32)
        
        return detected_pts
    
    def load_points(self):
        """
            load points by number of pattern, wvls, 2
            
        """
        # projector points
        proj_pts = np.zeros(shape=(self.total_px, 2)) # projector sensor plane pxs : # px, 2

        # 전체 패턴 개수 : 464개의 points / wvl + no filter : 11개 
        filter_zero_orders = np.zeros(shape=(self.total_px, len(self.wvls), 2))
        no_filter_zero_orders = np.zeros(shape=(self.total_px, 2))
        
        for i in range(len(os.listdir(self.pattern_npy_dir))):
            # projector pixel points
            proj_px = self.get_proj_px(os.path.join(self.pattern_npy_dir,"pattern_%05d.npy"%i))
            proj_pts[i] = proj_px
            
            # band pass filter zero orders
            detected_pts = self.get_detected_pts(i, proj_px) # m, wvl, 2 // 패턴의 개수 
            centroids = io.loadmat(os.path.join(self.points_dir, "pattern_%04d"%i, "no_fi_centroid.mat"))['centers']
            
            if len(centroids) == 0:
                no_filter_zero_orders[i] = np.zeros(shape=(2))
            else:
                no_filter_zero_orders[i] = centroids
                
            filter_zero_orders[i] = detected_pts[1].squeeze()
            
        return filter_zero_orders, no_filter_zero_orders
    
    def distortion(self, w, opt_param):
        """
            distort points by each wavelength
            
        """
        k1, k2, p1, p2, k3 = opt_param[w, 0], opt_param[w, 1], opt_param[w, 2], opt_param[w, 3], opt_param[w, 4]
        
        no_filter_ones = torch.ones(size=(self.total_px, 1), device=self.arg.device)
        filter_ones = torch.ones(size=(self.total_px, len(self.wvls), 1), device=self.arg.device)
        
        self.no_filter_zero_orders1 = torch.hstack((self.no_filter_zero_orders, no_filter_ones))
        self.filter_zero_orders1 = torch.concat((self.filter_zero_orders, filter_ones), dim = 2)

        no_filter_pts = torch.linalg.inv(self.cam_int)@self.no_filter_zero_orders1.transpose(1,0)
        filter_pts = torch.linalg.inv(self.cam_int)@self.filter_zero_orders1[:,w].transpose(1, 0)
        
        x_prime, y_prime = no_filter_pts[0], no_filter_pts[1]
        
        r = torch.sqrt(x_prime**2 + y_prime**2)        
        
        x_distorted = x_prime * (1 + k1*r**2 + k2 * r**4 + k3 * r**6) + (2*p1*x_prime*y_prime + p2*(r**2 + 2*x_prime**2))
        y_distorted = y_prime * (1 + k1*r**2 + k2 * r**4 + k3 * r**6) + (p1*(r**2 + 2*y_prime**2) + 2*p2*x_prime*y_prime)
        
        xy_distorted = torch.stack((x_distorted, y_distorted), dim = 1) # m, wvl, xyz1, px

        return xy_distorted, filter_pts[:2].transpose(1,0)
    
if __name__ == "__main__":    
    argument = Argument()
    arg = argument.parse()
    
    # args
    wvls = torch.tensor([430, 450, 480, 500, 520, 550, 580, 600, 620, 650, 660])
    positions = ["front", "mid", "back"]
    date = "0822"
    
    # optimized param initial values
    initial_value = torch.rand(len(wvls), 5)

    # parameters to be optimized
    opt_param = torch.tensor(initial_value, dtype= torch.float, requires_grad=True, device= arg.device)
    
    # loss ftn
    loss_f = torch.nn.L1Loss()
    losses = []

    # training args
    lr = 1e-2 # 1e-3
    decay_step = 400 # 1000
    epoch = 1100
    
    optimizer = torch.optim.Adam([opt_param], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.5)
    
    position_idx = 0
    
    for i in range(epoch):        
        for w in range(len(wvls)):
            if w == 0:
                xy_distorted, xy_ideal = PointDistortion(arg=arg, date=date, position=positions[position_idx]).distortion(w, opt_param)
                loss_wvl = loss_f(xy_distorted.to(torch.float32), xy_ideal.to(torch.float32))
                loss = loss_wvl
            else:
                xy_distorted, xy_ideal = PointDistortion(arg=arg, date=date, position=positions[position_idx]).distortion(w, opt_param)
                loss_wvl = loss_f(xy_distorted.to(torch.float32), xy_ideal.to(torch.float32))
                loss = loss + loss_wvl
                
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item() / (len(wvls)))
        optimizer.step()
        scheduler.step()

        if i % 30 == 0:
            print(f" Opt param value : {opt_param}, Epoch : {i}/{epoch}, Loss: {loss.item() / len(wvls)}, LR: {optimizer.param_groups[0]['lr']}")
        if (i % 1000 == 0) or (i == epoch-1): 
            np.save('./calibration/dg_calibration_method2/2023%s_data/opt_param/%sparam_%06d' %(date, positions[position_idx], i), opt_param.detach().cpu().numpy())

            