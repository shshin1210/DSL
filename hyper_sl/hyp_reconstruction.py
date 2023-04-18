import torch

def compute_hyp(arg, illum, cam_crf, I, batch_size):
    A = illum * cam_crf
    A = A.reshape(-1, batch_size * arg.num_train_px_per_iter, arg.wvl_num).permute(1,0,2)
    
    list_A = list(A)
    block = torch.block_diag(*list_A)

    x = torch.linalg.lstsq(block, I)
    x = x.solution
    
    return x

def diff_hyp(arg , x, x_gt, batch_size):
    diff = abs(x-x_gt).sum() / (arg.wvl_num * arg.num_train_px_per_iter * batch_size)
    
    return diff

def cal_A(arg, illum, cam_crf, batch_size, pixel_num):
    A = illum * cam_crf
    A = A.reshape(-1, batch_size * pixel_num, arg.wvl_num).permute(1,0,2)
    
    return A