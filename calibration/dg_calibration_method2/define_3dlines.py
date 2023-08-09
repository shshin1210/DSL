import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

import numpy as np
import matplotlib.pyplot as plt
from hyper_sl.utils.ArgParser import Argument
import torch

class Define3dLines():
    def __init__(self, arg, front_world_3d_pts, mid_world_3d_pts, back_world_3d_pts):
        # argument
        self.arg = arg
        self.wvls = np.arange(450, 660, 50)
        self.pts_num = (arg.proj_H // 10) * (arg.proj_W // 10)
        self.m_list = arg.m_list
        
        # 3d points : m, wvl, # pts, 3
        self.front_world_3d_pts = front_world_3d_pts
        self.mid_world_3d_pts = mid_world_3d_pts
        self.back_world_3d_pts = back_world_3d_pts
        
    def define3d_lines(self):
        # direction vector
        dir_vec = self.direction_vector()        
        
        # delete outliers of direction vector
        idx = (self.front_world_3d_pts[...,2] > 0.) * (self.mid_world_3d_pts[...,2] > 0.) * (self.back_world_3d_pts[...,2] > 0.)
        for i in range(arg.m_num*len(self.wvls)*self.pts_num):
            if idx[i] == False:
                dir_vec[i,:] = 0. # let outlier's direction vector be Zero

        dir_vec = dir_vec.reshape(self.arg.m_num, len(self.wvls), self.pts_num,3)
        
        return dir_vec
    
    def direction_vector(self):
        
        dir_vec = self.mean_method() # m, wvl, # pts, 3
        # dir_vec = self.opt_method() 
        
        return dir_vec
    
    
    def mean_method(self):
        points = np.stack(self.front_world_3d_pts, self.mid_world_3d_pts, self.back_world_3d_pts) # 3pts, m, wvl, # pts, xyz

        # Calculate the average of the last two points
        average_point = np.mean(points[1:], axis=0) # m, wvl, # pts, xyz

        # Calculate the direction vector
        direction_vector = average_point - points[0] # m, wvl, # pts, xyz

        # Normalize the direction vector
        normalized_direction = direction_vector / np.linalg.norm(direction_vector, axis = 3)[:,:,:,np.newaxis]

        return normalized_direction
    
    def opt_method(self):
        initial_point_value = torch.tensor(self.front_world_3d_pts)
        initial_dir_value = torch.tensor(self.back_world_3d_pts - self.front_world_3d_pts)

        initial_value = torch.hstack((initial_point_value,initial_dir_value))

        opt_param = torch.tensor(initial_value, dtype= torch.float, requires_grad=True, device= "cuda")

        # loss ftn
        loss = 0
        losses = []

        lr = 1e-3 # 1e-3
        decay_step = 2500 # 1000
        epoch = 15000

        num_pts = 3
        points = torch.stack((self.front_world_3d_pts, self.mid_world_3d_pts, self.back_world_3d_pts), device = "cuda")

        optimizer = torch.optim.Adam([opt_param], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.8)

        for i in range(epoch):
            loss = 0
            # distance between 3 points
            for k in range(3):
                dist = self.distance(opt_param[:3], points[k], opt_param[3:]) # p, q, dir
                loss += dist

            optimizer.zero_grad()
            loss.backward()
            
            losses.append(loss.item() / num_pts)
            optimizer.step()
            scheduler.step()

            if i % 1000 == 0:
                print(f" Opt param value : {opt_param}, Epoch : {i}/{epoch}, Loss: {loss.item() / num_pts}, LR: {optimizer.param_groups[0]['lr']}")
        
        return opt_param[:3], opt_param[3:]
    
    def distance(self, q, p, dir_vec):
        """
            p : point on the line
            q : 3d point
        
        """
        cross = torch.cross(q - p, dir_vec, dim = 3)
        distance = torch.norm(cross, p = 2, dim = 3) / torch.norm(dir_vec, p = 2, dim = 3)
        
        return distance    

    def visualization(self, order):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')

        ax.scatter(self.front_world_3d_pts[order,:,::10,0].flatten(), self.front_world_3d_pts[order,:,::10,1].flatten(), self.front_world_3d_pts[order,:,::10,2].flatten(), s =3.5)
        ax.scatter(self.mid_world_3d_pts[order,:,::10,0].flatten(), self.mid_world_3d_pts[order,:,::10,1].flatten(), self.mid_world_3d_pts[order,:,::10,2].flatten(), s= 3.5)
        ax.scatter(self.back_world_3d_pts[order,:,::10,0].flatten(), self.back_world_3d_pts[order,:,::10,1].flatten(), self.back_world_3d_pts[order,:,::10,2].flatten(), s= 3.5)

        ax.set_xlim([-0.15,0.15])
        ax.set_ylim([-0.1,0.1])
        ax.set_zlim([-0.,0.6])

        plt.title('%d order'%order)
        plt.show()

    def dir_visualization(self, dir_vec, order):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')

        scale = 1
        for i in range(0, 5*2303):
            start = [self.front_world_3d_pts[order,i,0], self.front_world_3d_pts[order,i,1], self.front_world_3d_pts[order,i,2]]

            X_d = [start[0], start[0] + scale*dir_vec[order,i,0]]
            Y_d = [start[1], start[1] + scale*dir_vec[order,i,1]]
            Z_d = [start[2], start[2] + scale*dir_vec[order,i,2]]
            
            ax.plot(X_d,Y_d,Z_d, color = 'red', linewidth = 1)
            
        ax.scatter(self.front_world_3d_pts[order,i,0].flatten(), self.front_world_3d_pts[order,i,1].flatten(), self.front_world_3d_pts[order,i,2].flatten(), s = 1.)
                
        ax.set_xlim([-0.15,0.15])
        ax.set_ylim([-0.1,0.1])
        ax.set_zlim([0.35,0.65])

        plt.title('%d order'%order)
        plt.show()

    
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    # wvl
    wvls = np.arange(450, 660, 50)
    pts_num = (arg.proj_H // 10) * (arg.proj_W // 10) -1
    
    # bring 3d points
    front_world_3d_pts = np.load('./front_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num, 3)
    mid_world_3d_pts = np.load('./mid_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num, 3)
    back_world_3d_pts = np.load('./back_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num, 3)

    # bring proj pts
    proj_pts = np.load('./proj_pts.npy')
    
    defining_3dlines = Define3dLines(arg, front_world_3d_pts, mid_world_3d_pts, back_world_3d_pts)
    
    # define direction vector
    dir_vec = defining_3dlines.define3d_lines()
    
    # visualization of second order
    defining_3dlines.visualization(2)

    # visualization of direction vector lines
    defining_3dlines.dir_visualization(dir_vec, 2)