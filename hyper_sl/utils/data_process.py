import numpy as np
import sys

sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging')
from hyper_sl.utils.ArgParser import Argument
from hyper_sl.utils import load_data
import torch.nn as nn
import cv2
    
    
def crop(data):
    
    argument = Argument()
    arg = argument.parse()

    start_x, start_y = 75,77
    res_x, res_y = arg.cam_W, arg.cam_H
    
    data = data[start_y: start_y + res_y, start_x:start_x+res_x]
    
    return data

def dilation(occ):
    # read the image
    img = occ.unsqueeze(dim = 2)
    
    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
    
    # dilate the image
    dilation = cv2.erode(img.numpy(), kernel, iterations=1) # 왜 erode인거징....
    
    return dilation

def to_patch(arg, data):
    data = data.reshape(-1, arg.cam_H, arg.cam_W, arg.illum_num,3).permute(0,3,4,1,2).reshape(-1,arg.illum_num *3, arg.cam_H, arg.cam_W) # B, N*3(C), H, W
    pad = nn.ReplicationPad2d((1))
    data_padded = pad(data) # B, N*3(C), H+2, W+2
    # N3_arr unfold
    uf = nn.Unfold(kernel_size=(3,3))
    data = uf(data_padded) # B, N*3(C) * 9, HxW
    data = data.permute(0,2,1).reshape(-1, arg.illum_num, 3, arg.patch_pixel_num).unsqueeze(dim =1)
    data = data.permute(0,4,1,2,3).reshape(-1, arg.illum_num, 3).unsqueeze(dim =1)

    return data


if __name__ == "__main__":

    argument = Argument()
    arg = argument.parse()
    
    occ = load_data.load_data(arg).load_occ(0)
    
    output = dilation(occ)
    print('end')
    
    