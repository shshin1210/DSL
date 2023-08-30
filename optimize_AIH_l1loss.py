import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

import hyper_sl.datatools as dtools 
from hyper_sl.image_formation_method2 import renderer
from hyper_sl.data import create_data_patch

import matplotlib.pyplot as plt


# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# print('cuda visible device count :',torch.cuda.device_count())
# print('current device number :', torch.cuda.current_device())

def optimizer_l1_loss(arg, b_dir, cam_crf):
    
    # arguments
    M = arg.cam_H * arg.cam_W
    R, C = arg.cam_H, arg.cam_W
    N = arg.illum_num
    W = arg.wvl_num
    device = arg.device
    
    # rendering function
    pixel_renderer = renderer.PixelRenderer(arg = arg)
    
    # create data function
    create_data = create_data_patch.createData
    
    # cam crf
    eval_dataset = dtools.pixelData(arg, train = True, eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False, real = True)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

    # render illumination data
    for i, data in enumerate(eval_loader):
        # real captured datas
        # N3_arr, cam_coord = data[0], data[1]
        # N3_arr, cam_coord = N3_arr.to(device = arg.device), cam_coord.to(device = arg.device)
        # np.save('./N3_arr_real.npy', N3_arr.detach().cpu().numpy())
        
        # to device         
        depth = torch.tensor(np.load("./checker_board_20230828.npy")[...,2].reshape(1,-1)).type(torch.float32).to(device=arg.device)

        # simulating checker board
        pixel_num = arg.cam_H * arg.cam_W
        random = False
        index = 0
            
        normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0).to(arg.device)
        normal = torch.zeros_like(normal)
        normal[:,2] = -1
        
        hyp = torch.tensor(np.load("./color_check_hyp_gt.npy")).type(torch.float32).to(device=arg.device).reshape(-1, arg.wvl_num)

        occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0).to(arg.device)
        occ = torch.ones_like(occ)
        
        cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0).to(arg.device)
        N3_arr, _, illum_data, _ = pixel_renderer.render(depth = depth, normal = normal, hyp = hyp, cam_coord = cam_coord, occ = occ, eval = True)
        np.save('./N3_arr_simulation_low_intensity_blur.npy', N3_arr.detach().cpu().numpy())
        
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

    # Captured image data (hdr)
    # b_dir = './hdr_step3.npy'
    # b = np.load(b_dir) / 65535.
    # b = b[:,:,:,::-1]
    # b = torch.tensor(b.copy(), device= device)
    b = N3_arr
    
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
        A_batch = (A[start_idx:end_idx])
        B_batch = (b[start_idx:end_idx])
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
    # np.save('./X_np_all_real.npy', X_np_all)
    np.save('./X_np_all_simulation_low_intensity_blur.npy', X_np_all)
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
    
    from hyper_sl.image_formation_method2 import camera
    
    cam_crf = camera.Camera(arg).get_CRF()
    cam_crf = torch.tensor(cam_crf, device= arg.device)

    b_dir = "./hdr_line_3.npy"
    optimizer_l1_loss(arg, cam_crf=cam_crf, b_dir= b_dir)