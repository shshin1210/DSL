import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer

import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
print('cuda visible device count :',torch.cuda.device_count())
print('current device number :', torch.cuda.current_device())

def optimizer_l1_loss(arg, b_dir, cam_crf):
    
    # arguments
    M = arg.cam_H * arg.cam_W
    R, C = arg.cam_H, arg.cam_W
    N = arg.illum_num
    W = arg.wvl_num
    device = arg.device
    
    # rendering function
    pixel_renderer = renderer.PixelRenderer(arg = arg)
    
    # cam crf
    eval_dataset = dtools.pixelData(arg, train = True, eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False, real = True)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

    # render illumination data
    for i, data in enumerate(eval_loader):
        # datas
        _, illum_data, cam_coord = data[0], data[1], data[2]
        
        # to device         
        depth = torch.tensor(np.load("./calibration/color_checker_depth_0508.npy")[...,2].reshape(1,-1)).type(torch.float32)

        _, _, illum_data, _ = pixel_renderer.render(depth, None, None, None, cam_coord, None, True)
    
    illum_data = illum_data.to(device)
    
    # Illum data
    A = illum_data[0].unsqueeze(dim = 2)
    cam_crf = cam_crf.unsqueeze(dim =0).unsqueeze(dim = 0)
    cam_crf = cam_crf.permute(0,1,3,2)
    A = A * cam_crf
    
    # A = illum_data[0].unsqueeze(dim = 1)
    # cam_crf = cam_crf.unsqueeze(dim =0).unsqueeze(dim = 0)
    # cam_crf = cam_crf.permute(0,3,1,2)
    # A = A * cam_crf

    # Captured image data
    # b = np.load(b_dir) / 65535.
    b = np.load(b_dir)
    # b = b[:,:,:,::-1]
    b = torch.tensor(b, device= device)
    
    # optimize with l1 loss
    # Reshape to make M, ...
    r, c = 290, 445

    A = A.reshape(R, C, 1, 3*N, W)
    b = b.reshape(R, C, 1, 3*N, 1)

    A1 = A[:r,:c]
    A2 = A[:r,c:]
    A3 = A[r:,:c]
    A4 = A[r:,c:]

    b1 = b[:r,:c]
    b2 = b[:r,c:]
    b3 = b[r:,:c]
    b4 = b[r:,c:]

    A_list = [A1,A2,A3,A4]
    b_list = [b1,b2,b3,b4] 
    
    batch_size = 100000
    num_iter = 5000

    loss_f = torch.nn.L1Loss()
    losses = []
    X_np_all = torch.zeros(R, C, 1, W, 1)

    # define initial learning rate and decay step
    lr = 1
    decay_step = 500

    # training loop over batches
    for batch_idx in range(len(A_list)):
        A_batch = (A_list[batch_idx]).to(device).reshape(r*c,1, 3*N, W)
        B_batch = (b_list[batch_idx]).to(device).reshape(r*c,1, 3*N, 1)
        X_est = torch.randn(r*c, 1, W, 1, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([X_est], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.5)

        optimizer.zero_grad()
        for i in range(num_iter):
            loss = loss_f(A_batch @ X_est, B_batch)
            X_est_reshape = X_est.reshape(r,c,W).unsqueeze(dim = 0).permute(0,3,1,2)
            loss_tv = total_variation_loss_l1(X_est_reshape, 0.1)
            loss_spec = total_variation_loss_l2_spectrum(X_est_reshape, 0.1)
            total_loss = loss + loss_tv + loss_spec
            
            total_loss.backward()
            losses.append(total_loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print(f"Batch {batch_idx + 1}/{len(A_list)}, Iteration {i}/{num_iter}, Loss: {loss.item()}, TV Loss: {loss_tv.item()}, Spec Loss: {loss_spec.item()},  LR: {optimizer.param_groups[0]['lr']}")

        if batch_idx == 0:
            X_np_all[:r,:c]= X_est.detach().cpu().reshape(r,c,1,W,1)
        elif batch_idx == 1:
            X_np_all[:r,c:]= X_est.detach().cpu().reshape(r,c,1,W,1)
        elif batch_idx == 2:
            X_np_all[r:,:c]= X_est.detach().cpu().reshape(r,c,1,W,1)
        else:
            X_np_all[r:,c:]= X_est.detach().cpu().reshape(r,c,1,W,1)

    X_np_all = X_np_all.numpy()
    np.save('./X_np_all_step_tv_wo_spec', X_np_all)

    # plot losses over time
    plt.figure(figsize=(15,10))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss over time")
    plt.xlabel("Iteration")
    plt.ylabel("L1 Loss")

    plt.subplot(1, 2, 2)
    plt.semilogy(losses)
    plt.title("Log Loss over time")
    plt.xlabel("Iteration")
    plt.ylabel("L1 Loss (log scale)")

    plt.show()

def total_variation_loss_l2(img, weight): 
    bs_img, c_img, h_img, w_img = img.size() 
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum() 
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum() 
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def total_variation_loss_l1(img, weight): 
    bs_img, c_img, h_img, w_img = img.size() 
    tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum() 
    tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum() 
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def total_variation_loss_l2_spectrum(img, weight): 
    bs_img, c_img, h_img, w_img = img.size() 
    tv_s = torch.pow(img[:,1:,:,:]-img[:,:-1,:,:], 2).sum()
    return weight*(tv_s)/(bs_img*c_img*h_img*w_img)

if __name__ == "__main__":
    
    argument = Argument()
    arg = argument.parse()
    
    from hyper_sl.image_formation import camera
    
    cam_crf = camera.Camera(arg).get_CRF()
    cam_crf = torch.tensor(cam_crf, device= arg.device)

    b_dir = "./line_3_saturate_0510.npy"
    optimizer_l1_loss(arg, cam_crf=cam_crf, b_dir= b_dir)