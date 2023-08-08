import cv2, os, sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')

import numpy as np
import matplotlib.pyplot as plt
from hyper_sl.utils.ArgParser import Argument

def visualization(front_world_3d_pts, back_world_3d_pts, order):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.scatter(front_world_3d_pts[order,:,::10,0].flatten(), front_world_3d_pts[order,:,::10,1].flatten(), front_world_3d_pts[order,:,::10,2].flatten(), s =3.5)
    ax.scatter(back_world_3d_pts[order,:,::10,0].flatten(), back_world_3d_pts[order,:,::10,1].flatten(), back_world_3d_pts[order,:,::10,2].flatten(), s= 3.5)

    ax.set_xlim([-0.15,0.15])
    ax.set_ylim([-0.1,0.1])
    ax.set_zlim([-0.,0.6])

    plt.title('%d order'%order)
    plt.show()

def dir_visualization(dir_vec, front_world_3d_pts, order):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    scale = 1
    for i in range(0, 5*2303):
        # if (dir_vec[order, i, 2] < 0.) or (front_world_3d_pts[order,i,2] < 0.):
        #     continue
        # else:
        start = [front_world_3d_pts[order,i,0], front_world_3d_pts[order,i,1], front_world_3d_pts[order,i,2]]

        X_d = [start[0], start[0] + scale*dir_vec[order,i,0]]
        Y_d = [start[1], start[1] + scale*dir_vec[order,i,1]]
        Z_d = [start[2], start[2] + scale*dir_vec[order,i,2]]
        
        ax.plot(X_d,Y_d,Z_d, color = 'red', linewidth = 1)
        
    ax.scatter(front_world_3d_pts[order,i,0].flatten(), front_world_3d_pts[order,i,1].flatten(), front_world_3d_pts[order,i,2].flatten(), s = 1.)
            
    ax.set_xlim([-0.15,0.15])
    ax.set_ylim([-0.1,0.1])
    ax.set_zlim([0.35,0.65])

    plt.title('%d order'%order)
    plt.show()

    
def main(arg):
    
    # wvl
    wvls = np.arange(450, 660, 50)
    pts_num = (arg.proj_H // 10) * (arg.proj_W // 10) -1
    
    # bring 3d points
    front_world_3d_pts = np.load('./front_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num, 3)
    back_world_3d_pts = np.load('./back_world_3d_pts.npy').reshape(arg.m_num, len(wvls), pts_num, 3)

    # bring proj pts
    proj_pts = np.load('./proj_pts.npy')
    
    # visualization of second order
    # visualization(front_world_3d_pts, back_world_3d_pts, 2)

    front_world_3d_pts_reshape = front_world_3d_pts.reshape(-1, 3)
    back_world_3d_pts_reshape = back_world_3d_pts.reshape(-1, 3)

    # delete outliers
    dir_vec_reshape = back_world_3d_pts_reshape - front_world_3d_pts_reshape
    idx = (front_world_3d_pts_reshape[...,2] > 0.) * ( back_world_3d_pts_reshape[...,2] > 0.)
    
    for i in range(3*5*pts_num):
        if idx[i] == False:
            dir_vec_reshape[i,:] = 0.
    
    dir_vec = dir_vec_reshape.reshape(3,5*2303,3)
    front_world_3d_pts = front_world_3d_pts.reshape(3, 5*2303,3)
    
    # visualization of direction vector lines
    # dir_visualization(dir_vec, front_world_3d_pts, 2)

    # define lines
    
if __name__ == "__main__":
    argument = Argument()
    arg = argument.parse()
    
    main(arg)