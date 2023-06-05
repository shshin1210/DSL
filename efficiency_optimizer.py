import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument
from hyper_sl.data import create_data_patch
import hyper_sl.datatools as dtools 
from hyper_sl.image_formation.rendering_prac import renderer_opt

import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('cuda visible device count :',torch.cuda.device_count())
print('current device number :', torch.cuda.current_device())

def optimizer_l1_loss(arg, dg_efficiency):
    
    # arguments
    pixel_num = arg.cam_H * arg.cam_W
    random = False
    index = 0
                
    # rendering function
    pixel_renderer = renderer_opt.PixelRenderer(arg = arg, dg_efficiency = dg_efficiency)
    
    # cam crf
    eval_dataset = dtools.pixelData(arg, train = False,eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

    # render illumination data
    for i, data in enumerate(eval_loader):
        # datas
        create_data = create_data_patch.createData
        depth = torch.tensor(np.load("./calibration/spectralon_depth_0510.npy")[...,2].reshape(1,-1), device = arg.device).type(torch.float32)
        normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
        normal = torch.zeros_like(normal)
        normal[:,2] = -1.
        hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
        hyp = torch.ones_like(hyp)
        hyp[:] = 0.9
        occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
        occ = torch.ones_like(occ)
        cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0).to(device=arg.device)

        # rendered image
        N3_arr, _, _, _  = pixel_renderer.render(depth = depth, 
                                        normal = normal, hyp = hyp, occ = occ, 
                                        cam_coord = cam_coord, eval = False)

        return N3_arr

if __name__ == "__main__":
    
    argument = Argument()
    arg = argument.parse()
        
    opt_param = torch.randn(size = (3,25), dtype= torch.float, requires_grad=True,device=arg.device)

    lr = 1e-3
    decay_step = 1000

    epoch = 5000
    loss_f = torch.nn.L1Loss()
    losses = []
    
    optimizer = torch.optim.Adam([opt_param], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma = 0.7)

    for i in range(epoch):        
            
        N3_arr = optimizer_l1_loss(arg, dg_efficiency= opt_param)

        loss = loss_f(N3_arr, 'Real Image')
                
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        
        scheduler.step()

        if i % 100 == 0:
            print(f" Opt param value : {opt_param}, Epoch : {i}/{epoch}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
            
    plt.figure()
    plt.plot(losses)
    plt.savefig('./loss_ftn.png')