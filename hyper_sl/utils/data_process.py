import numpy as np
from hyper_sl.utils.ArgParser import Argument
import torch.nn as nn

def crop(data):
    
    argument = Argument()
    arg = argument.parse()

    start_x, start_y = 75,77
    res_x, res_y = arg.cam_W, arg.cam_H
    
    data = data[start_y: start_y + res_y, start_x:start_x+res_x]
    
    return data

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