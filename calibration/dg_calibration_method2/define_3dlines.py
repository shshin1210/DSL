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
        # self.wvls = np.arange(450, 660, 50)
        self.wvls = np.array([430, 450, 480, 500, 520, 550, 580, 600, 620, 650, 660])
        self.pts_num = (arg.proj_H // 10) * (arg.proj_W // 10)
        self.m_list = arg.m_list
        
        # 3d points : m, wvl, # pts, 3
        self.front_world_3d_pts = front_world_3d_pts
        self.mid_world_3d_pts = mid_world_3d_pts
        self.back_world_3d_pts = back_world_3d_pts
        
    def define3d_lines(self):
        """
            Final step to define direction vector : delete outliers
            Outliers : (0, 0) for camera (u, v) detected points
                        has negative value for z values
                        
            returns : direction vector & start point
        """
        
        # direction vector
        start_pts, dir_vec = self.direction_vector()
        
        dir_vec = dir_vec.reshape(-1, 3).detach().cpu().numpy()
        start_pts = start_pts.detach().cpu().numpy()

        front_world_3d_pts = self.front_world_3d_pts.reshape(-1, 3)
        back_world_3d_pts = self.back_world_3d_pts.reshape(-1, 3)
        mid_world_3d_pts = self.mid_world_3d_pts.reshape(-1, 3)
         
        # delete outliers of direction vector (delete z points under 0.)
        idx = (front_world_3d_pts[...,2] > 0.) * (back_world_3d_pts[...,2] > 0.) * (mid_world_3d_pts[...,2] > 0.)
        for i in range(self.arg.m_num*len(self.wvls)*self.pts_num):
            if idx[i] == False:
                dir_vec[i,:] = 0. # let outlier's direction vector be Zero

        dir_vec = dir_vec.reshape(self.arg.m_num, len(self.wvls), self.pts_num, 3)

        return dir_vec, start_pts
    
    def direction_vector(self):
        """
            choose between 2 direction vector method
        """
        
        # dir_vec = self.mean_method() # m, wvl, # pts, 3
        start_pts, dir_vec = self.opt_method() 
        
        return start_pts, dir_vec
    
    def mean_method(self):
        """
            calculate direction vetor with mean method
            starting point : front spectralon point
            mean of 2 points on middle and back spectralon
            
            the direction vector is the direction between front & mean point
            
            return : direction vector
        """
        points = np.stack((self.front_world_3d_pts, self.mid_world_3d_pts, self.back_world_3d_pts)) # 3pts, m, wvl, # pts, xyz

        # Calculate the average of the last two points
        average_point = np.mean(points[1:], axis=0) # m, wvl, # pts, xyz

        # Calculate the direction vector
        direction_vector = average_point - points[0] # m, wvl, # pts, xyz

        # Normalize the direction vector
        normalized_direction = direction_vector / np.linalg.norm(direction_vector, axis = 3)[:,:,:,np.newaxis]

        return normalized_direction
    
    def opt_method(self):
        """
            Optimize the direction vector of three points
            Three points : front / middle / back spectralon 
            
            return : direction vector, starting point
        
        """
        self.front_world_3d_pts = torch.tensor(self.front_world_3d_pts)
        self.mid_world_3d_pts = torch.tensor(self.mid_world_3d_pts)
        self.back_world_3d_pts = torch.tensor(self.back_world_3d_pts)

        initial_point_value = torch.tensor(self.front_world_3d_pts)
        initial_dir_value = torch.tensor(self.back_world_3d_pts - self.front_world_3d_pts + 1e-4)

        initial_value = torch.cat((initial_point_value,initial_dir_value), dim = 3).type(torch.float)

        opt_param = torch.tensor(initial_value, dtype= torch.float, requires_grad=True, device= "cuda")

        # loss ftn
        loss = 0
        losses = []

        lr = 1e-3 # 1e-3
        decay_step = 2500 # 1000
        epoch = 15000

        num_pts = 3
        points = torch.stack((self.front_world_3d_pts, self.mid_world_3d_pts, self.back_world_3d_pts)).to(device="cuda").type(torch.float)

        optimizer = torch.optim.Adam([opt_param], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.8)

        for i in range(epoch):
            loss = 0
            # distance between 3 points
            for k in range(3):
                dist = self.distance(opt_param[...,:3], points[k], opt_param[...,3:]) # p, q, dir
                loss += abs(dist)

            loss = loss.sum()
            optimizer.zero_grad()
            loss.backward()
            
            losses.append(loss.item() / (num_pts*self.arg.m_num*len(self.wvls)*self.pts_num))
            optimizer.step()
            scheduler.step()

            if i % 1000 == 0:
                print(f" Opt param value : {opt_param}, Epoch : {i}/{epoch}, Loss: {loss.item() / (num_pts*self.arg.m_num*len(self.wvls)*self.pts_num)}, LR: {optimizer.param_groups[0]['lr']}")
        
        return opt_param[...,:3], opt_param[...,3:]
    
    def distance(self, q, p, dir_vec):
        """
            distance between a single point and a line
            
            p : point on the line
            q : 3d point
        
        """
        cross = torch.cross(q - p, dir_vec, dim = 3)
        distance = torch.norm(cross, p = 2, dim = 3) / torch.norm(dir_vec, p = 2, dim = 3)
        
        return distance    

    def visualization(self, order):
        """
            visualization of 3d points on three different placed spectralons
            
        """
        fig = plt.figure()
        ax = plt.axes(projection = '3d')

        ax.scatter(self.front_world_3d_pts[order,:,::10,0].flatten(), self.front_world_3d_pts[order,:,::10,1].flatten(), self.front_world_3d_pts[order,:,::10,2].flatten(), s =3.5)
        ax.scatter(self.mid_world_3d_pts[order,:,::10,0].flatten(), self.mid_world_3d_pts[order,:,::10,1].flatten(), self.mid_world_3d_pts[order,:,::10,2].flatten(), s= 3.5)
        ax.scatter(self.back_world_3d_pts[order,:,::10,0].flatten(), self.back_world_3d_pts[order,:,::10,1].flatten(), self.back_world_3d_pts[order,:,::10,2].flatten(), s= 3.5)

        ax.set_xlim([-0.15,0.15])
        ax.set_ylim([-0.1,0.1])
        ax.set_zlim([-0.6,0.6])

        plt.title('%d order'%order)
        plt.show()

    def dir_visualization(self, dir_vec, start_pts, order_idx, wvl_idx):
        """
            Visualization of direction vector & start point : line
            and points on three different placed spectralon 
        """
        
        fig = plt.figure()
        ax = plt.axes(projection = '3d')

        m = order_idx
        wvl = wvl_idx
                
        for k in range(0, self.pts_num, 20):
            
            ax.scatter(start_pts[m, wvl, k, 0], start_pts[m, wvl, k, 1], start_pts[m, wvl, k, 2], c = 'cyan', s = 5)
            ax.scatter(self.front_world_3d_pts[m, wvl, k, 0], self.front_world_3d_pts[m, wvl, k,1], self.front_world_3d_pts[m, wvl, k, 2] , c = 'blue', s =5)
            ax.scatter(self.mid_world_3d_pts[m, wvl, k, 0], self.mid_world_3d_pts[m, wvl, k, 1], self.mid_world_3d_pts[m, wvl, k, 2] , c = 'green', s =5)
            ax.scatter(self.back_world_3d_pts[m, wvl, k, 0], self.back_world_3d_pts[m, wvl, k, 1], self.back_world_3d_pts[m, wvl, k, 2] , c = 'purple', s =5)

            scale = 1
            X_d = [start_pts[m, wvl, k, 0], start_pts[m, wvl, k, 0] + scale* dir_vec[m, wvl, k, 0]]
            Y_d = [start_pts[m, wvl, k, 1], start_pts[m, wvl, k, 1] + scale* dir_vec[m, wvl, k, 1]]
            Z_d = [start_pts[m, wvl, k, 2], start_pts[m, wvl, k, 2] + scale* dir_vec[m, wvl, k, 2]]

            ax.plot(X_d,Y_d,Z_d, color = 'red', linewidth = 1)


            scale = -0.5
            X_d = [start_pts[m, wvl, k, 0], start_pts[m, wvl, k, 0] + scale* dir_vec[m, wvl, k, 0]]
            Y_d = [start_pts[m, wvl, k, 1], start_pts[m, wvl, k, 1] + scale* dir_vec[m, wvl, k, 1]]
            Z_d = [start_pts[m, wvl, k, 2], start_pts[m, wvl, k, 2] + scale* dir_vec[m, wvl, k, 2]]
            
            ax.plot(X_d,Y_d,Z_d, color = 'red', linewidth = 1)

        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        ax.set_xlim([-0.15,0.15])
        ax.set_ylim([-0.1,0.1])
        ax.set_zlim([0.35,0.8])

        plt.title('%d order %dnm wvl'%(self.m_list[m],self.wvls[wvl]))
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