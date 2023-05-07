import torch
import sys

def normalization(arg, data):   
    """
        normalize projector coord xy
    """
    
    # x_gt_min, x_gt_max = torch.tensor([-0.0024 - 1e-4], device= arg.device), torch.tensor([0.0027 + 1e-4], device= arg.device)
    # y_gt_min, y_gt_max = torch.tensor([-0.0033 - 1e-4], device= arg.device), torch.tensor([0.0005 + 1e-4], device= arg.device)
    
    x_gt_min, x_gt_max = torch.tensor([-0.0026 - 1e-4], device= arg.device), torch.tensor([0.0024 + 1e-4], device= arg.device)
    y_gt_min, y_gt_max = torch.tensor([-0.0027 - 1e-4], device= arg.device), torch.tensor([7.7517e-05 + 1e-4], device= arg.device)
    
    
    x_gt_minmax = (data[...,0] - x_gt_min.unsqueeze(dim = 1)) / (x_gt_max.unsqueeze(dim = 1) - x_gt_min.unsqueeze(dim = 1)) 
    y_gt_minmax = (data[...,1] - y_gt_min.unsqueeze(dim = 1)) / (y_gt_max.unsqueeze(dim = 1) - y_gt_min.unsqueeze(dim = 1))
    
    xy_proj_mm = torch.stack((x_gt_minmax, y_gt_minmax), dim = 2)
        
    return xy_proj_mm

def un_normalization(data):
    """
        unnormalize projector coord xy
    """
    
    # x_min, x_max = -0.0024 - 1e-4, 0.0027 + 1e-4
    # y_min, y_max = -0.0033 - 1e-4, 0.0005 + 1e-4
    
    x_min, x_max = -0.0026 - 1e-4, 0.0024 + 1e-4
    y_min, y_max = -0.0027 - 1e-4, 7.7517e-05 + 1e-4
    
    
    data_unnorm_x = data[...,0] * (x_max - x_min) + x_min
    data_unnorm_y = data[...,1] * (y_max - y_min) + y_min

    data_unnorm = torch.stack((data_unnorm_x, data_unnorm_y), dim = 2)
    
    return data_unnorm    

def N3_normalize(N3_arr, illum_num):
    """
        normalization of N3_arr
    """
    
    N3_arr_r = N3_arr.reshape(-1, illum_num*3)
    N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
    N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
    N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)

    return N3_arr_normalized
