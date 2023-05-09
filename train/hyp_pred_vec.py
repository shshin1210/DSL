import hyper_sl.datatools as dtools 
from torch.utils.data import DataLoader
import torch
from hyper_sl.utils.ArgParser import Argument
import numpy as np
import os
import scipy.sparse as sparse
from scipy.sparse import linalg

argument = Argument()
arg = argument.parse()

cam_crf = torch.tensor(np.load(os.path.join(arg.camera_response,'CRF_cam.npy')), device= arg.device).T

eval_dataset = dtools.hypData(arg, train = False,eval = True, pixel_num = arg.cam_H*arg.cam_W, bring_data = arg.load_dataset, random = False)
eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

x_all = torch.zeros(arg.cam_H*arg.cam_H, 29)
x_gt_all = torch.zeros(arg.cam_H*arg.cam_H, 29)
        
def cal_A(illum, cam_crf):
    A = illum * cam_crf
    
    return A

for i, data in enumerate(eval_loader):
    # cam coord, gt xyz return
    
    N3_arr_data = data[0].to(arg.device) # B, # pixel, N, 3
    illum_data = data[3].to(arg.device) # B, # pixel, N, 29 / illumination
    hyp_gt_data = data[4].to(arg.device) # B, # pixel, 29
    shading = data[5].to(arg.device) # B, 3(m), 29(wvl), # pixel

    # reshape input
    illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2) # N, M, 29
    illum = illum.unsqueeze(dim = 1)  # N, 1, M, 29
    cam_crf = cam_crf[None,:,:].unsqueeze(dim = 2) # 1, 3, 1, 29
    
    # reshape I_c
    N3_arr = N3_arr_data.reshape(-1, arg.illum_num, 3) # M, N, 3
    I = N3_arr.reshape(-1,arg.illum_num * 3) # N3, M
    I = I.reshape(-1,1) # N3M, 1
    
    # ground truth
    hyp_gt = hyp_gt_data.reshape(-1, arg.wvl_num) # M, 29
    hyp_gt = hyp_gt.reshape(-1, 1) # 29M, 1

    # x_gt
    shading_term = shading[0,0,:,:].reshape(-1,1) # 29M, 1
    x_gt = shading_term*hyp_gt # 29M, 1
    
    # A
    A = cal_A(illum, cam_crf) # N, 3, M, 29
    # A = A.reshape(-1, pixel_num, arg.wvl_num).permute(1,0,2).to(arg.device) # 3N, M, 29  
    A = A.reshape(-1, arg.cam_H*arg.cam_H, arg.wvl_num).permute(1,0,2)
    
    # list_A = list(A)
    list_A = list(A.detach().cpu().numpy())
    # block = scipy.linalg.block_diag(*list_A)
    block = sparse.block_diag(mats = list_A)
    
    # block = sparse.csr_matrix(scipy.linalg.block_diag(*list_A))
    # block = sparse.csr_matrix(block)
    
    # detach
    I = I.detach().cpu().numpy()
    x_gt = x_gt.detach().cpu().numpy()
    # I = I.numpy()
    # x_gt = x_gt.numpy()
    
    x = linalg.lsqr(block, I)[0] # 400, 29
    
    # x = torch.linalg.lstsq(A, I)
    # x = x.solution
    
    # I_sim = A@x_gt # 378 (126 * 3), 1
    I_sim = block @ x_gt
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(I_sim.data.cpu())
    plt.plot(I.data.cpu())
    plt.legend(['I_sim', 'I'])
    plt.savefig('./tmp.png')
    
    x_all = x.reshape(-1, arg.wvl_num)
    x_gt_all = x_gt.reshape(-1, arg.wvl_num)
    
    diff = abs(x-x_gt).sum() / (arg.cam_H*arg.cam_H * arg.wvl_num)
    
    torch.cuda.empty_cache()
print('end')