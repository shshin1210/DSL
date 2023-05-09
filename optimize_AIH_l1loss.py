import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer

import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
        N3_arr, illum_data, cam_coord = data[0], data[1], data[2]
        
        # to device         
        depth = torch.tensor(np.load("./calibration/color_checker_depth_0508.npy")[...,2].reshape(1,-1)).type(torch.float32)

        _, xy_proj_real_norm, illum_data, _ = pixel_renderer.render(depth, None, None, None, cam_coord, None, True)
                        
        illum_data = illum_data.to(arg.device)
    
    # Illum data
    A = illum_data[0].detach().cpu().numpy()
    
    # Make A array
    A = np.expand_dims(A, axis=2)
    cam_crf = np.expand_dims(np.expand_dims(cam_crf, axis =0), axis=0)
    cam_crf = cam_crf.transpose(0,1,3,2)
    A = A * cam_crf

    # Captured image data
    b = np.load(b_dir) / 65535.
    b = b[:,:,:,::-1]
    
    # optimize with l1 loss
    # Reshape to make M, ...
    A = A.reshape(R*C, 1, 3*N, W)
    b = b.reshape(R*C, 1, 3*N, 1)    
        
    batch_size = 100000
    num_iter = 5000
    num_batches = int(np.ceil(M / batch_size))
    loss_f = torch.nn.L1Loss()
    losses = []
    X_np_all = torch.zeros(M, 1, W, 1)

    # define initial learning rate and decay step
    lr = 0.5
    decay_step = 800

    # training loop over batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, M)
        batch_size_ = end_idx - start_idx
        A_batch = torch.from_numpy(A[start_idx:end_idx]).to(device)
        B_batch = torch.from_numpy(b[start_idx:end_idx]).to(device)
        X_est = torch.randn(batch_size_, 1, W, 1, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([X_est], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.5)

        optimizer.zero_grad()
        for i in range(num_iter):
            loss = loss_f(A_batch @ X_est, B_batch)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, Iteration {i}/{num_iter}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

        X_np_all[start_idx:end_idx] = X_est.detach().cpu()

    X_np_all = X_np_all.numpy()
    np.save('X_np_all_%s'%b_dir, X_np_all)

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


if __name__ == "__main__":
    
    argument = Argument()
    arg = argument.parse()
    
    from hyper_sl.image_formation import camera
    
    cam_crf = camera.Camera(arg).get_CRF()

    b_dir = "./hdr_step3.npy"
    optimizer_l1_loss(arg, cam_crf=cam_crf, b_dir= b_dir)