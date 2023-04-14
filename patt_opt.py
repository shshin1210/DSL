import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

from hyper_sl.mlp import MLP
import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer
from hyper_sl.hyp_reconstruction import compute_hyp, diff_hyp

from torch.utils.tensorboard import SummaryWriter

def train(arg, epochs, cam_crf):
    writer = SummaryWriter(log_dir=arg.log_dir)
    # writer = SummaryWriter(log_dir='./logs_depth')
    
    train_dataset = dtools.pixelData(arg, train = True, eval = False, pixel_num = arg.num_train_px_per_iter)
    train_loader = DataLoader(train_dataset, batch_size = arg.batch_size_train, shuffle = True)

    test_dataset = dtools.pixelData(arg, train = False,eval = False, pixel_num = arg.num_train_px_per_iter)
    test_loader = DataLoader(test_dataset, batch_size = arg.batch_size_test, shuffle = True)

    eval_dataset = dtools.pixelData(arg, train = False,eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

    # bring model MLP
    model = MLP(input_dim = arg.illum_num*3, output_dim = 1).to(device=arg.device)
    model_path = arg.model_dir
    model_num = 'model_coord_%05d.pth'%(arg.model_num)
    model.load_state_dict(torch.load(os.path.join(model_path, model_num)))
    
    # optimizer, schedular, loss function
    optimizer = torch.optim.Adam(list(model.parameters()), lr= arg.model_lr)
    scheduler = torch.optim.lr_scheduler.StepLR((optimizer), step_size= arg.model_step_size, gamma= arg.model_gamma)
    
    # illumination
    illum_opt = torch.nn.Parameter(arg.illums) # num, H, W, 3
    # illum_data = os.path.join(arg.illum_data_dir, 'illum_data.npy')
    optimizer_illum = torch.optim.Adam([illum_opt], lr = arg.illum_lr)
    scheduler_illum = torch.optim.lr_scheduler.StepLR((optimizer_illum), step_size= arg.illum_step_size, gamma= arg.illum_gamma)
    
    # loss ftn
    loss_fn = torch.nn.L1Loss()
    loss_fn.requires_grad_ = True
    
    # rendering function
    pixel_renderer = renderer.PixelRenderer(arg = arg)
    
    # cam crf
    cam_crf = cam_crf[None,:,:].unsqueeze(dim = 2)
    
    for epoch in range(epochs):
        model.train()
        
        losses_total = []
        losses_depth = []
        losses_hyp = []
        
        total_iter = 0
        
        for i, data in enumerate(train_loader):
            # datas
            depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
            
            # illum render함수에 넣기
            
            # image formation for # of pixels
            N3_arr, gt_xy, _, illum_data, shading  = pixel_renderer.render(depth = depth, 
                                                        normal = normal, hyp = hyp, occ = occ, 
                                                        cam_coord = cam_coord, eval = False, illum_opt=illum_opt)
            # batch size
            batch_size = N3_arr.shape[0]
            
            # to device
            # B, # pixel, N, 3 / B, # pixel, 2 / B, # pixel, N, 29 / B, # pixel, 29 / B, 3(m), 29(wvl), # pixel
            N3_arr, gt_x, illum_data, hyp, shading = N3_arr.to(arg.device), gt_xy[...,0].to(arg.device), illum_data.to(arg.device), hyp.to(arg.device), shading.to(arg.device)
            
            # reshape
            N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
            I = N3_arr.reshape(-1, arg.illum_num * 3)
            
            N3_arr = N3_arr.unsqueeze(dim = 1) 
            gt_x = gt_x.reshape(-1,1)
            
            # hyp datas reshape
            illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
            I = I.reshape(-1, 1)
            hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
            hyp = hyp.reshape(-1, 1)
            
            # hyp gt data
            shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1,1) # 29M, 1
            x_gt = shading_term * hyp
            
            # normalization of N3_arr
            N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
            N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
            N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
            N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)

            # model coord
            pred_xy = model(N3_arr_normalized)
            loss_depth = loss_fn(gt_x, pred_xy)
            
            # hyp lstsq
            x = compute_hyp(arg, illum, cam_crf, I, batch_size)
            loss_hyp = diff_hyp(arg, x, x_gt, batch_size)

            # save last epoch training set
            if epoch == arg.epoch_num -1 :
                torch.save(N3_arr, os.path.join(arg.output_dir, f'N3_arr_{epoch}.pt'))
                torch.save(pred_xy, os.path.join(arg.output_dir, f'pred_xy_{epoch}.pt'))
            
            losses_depth.append(loss_depth.item())
            losses_hyp.append(loss_hyp)
            
            # loss = (loss_depth / (1/arg.proj_H)) * arg.weight_depth
            # loss = (loss_depth / (1/arg.proj_H)) * arg.weight_depth + loss_hyp * arg.weight_hyp
            loss = (loss_depth * 10) * arg.weight_depth + loss_hyp * arg.weight_hyp
            losses_total.append(loss.item())
            
            optimizer.zero_grad()
            optimizer_illum.zero_grad()            
            loss.backward()
            optimizer.step()
            optimizer_illum.step()
            total_iter += 1
            
        scheduler.step()
        scheduler_illum.step()
        
        epoch_total_loss = (sum(losses_total)/ total_iter)
        epoch_train_depth_px = (sum(losses_depth)/total_iter) / (1/arg.proj_H)
        epoch_train_depth = (sum(losses_depth)/total_iter) * 10 
        epoch_train_hyp = (sum(losses_hyp)/total_iter)
        
        print("{%dth epoch} Train depth Loss: "%(epoch), epoch_train_depth_px)
        writer.add_scalars("Training Depth", {'depth_loss' : epoch_train_depth, 'weight_loss': epoch_total_loss}, epoch)
        # writer.add_scalar("Training Depth", epoch_train_depth_px, epoch)

        print("{%dth epoch} Train Hyp Error: "%(epoch), epoch_train_hyp)
        writer.add_scalars("Training Hyp", {'hyp_loss' : epoch_train_hyp, 'weight_loss' : epoch_total_loss}, epoch)
        
        # evaluation
        model.eval()
        
        with torch.no_grad():
            
            losses_total = []
            losses_depth = []
            losses_hyp = []
            total_iter = 0
            
            for i, data in enumerate(test_loader):   
                # datas
                depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
                print(f'rendering for {depth.shape[0]} scenes at {i}-th iteration')
                # image formation
                N3_arr, gt_xy, _ , illum_data, shading = pixel_renderer.render(depth = depth, 
                                                            normal = normal, hyp = hyp, occ = occ, 
                                                            cam_coord = cam_coord, eval = False, illum_opt=illum_opt)
                
                # batch size
                batch_size = N3_arr.shape[0]
                
                # to device
                # B, # pixel, N, 3 / B, # pixel, 2 / B, # pixel, N, 29 / B, # pixel, 29 / B, 3(m), 29(wvl), # pixel
                N3_arr, gt_x, illum_data, hyp, shading = N3_arr.to(arg.device), gt_xy[...,0].to(arg.device), illum_data.to(arg.device), hyp.to(arg.device), shading.to(arg.device)                

                # reshape
                N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
                I = N3_arr.reshape(-1, arg.illum_num * 3)
                
                N3_arr = N3_arr.unsqueeze(dim = 1) 
                gt_x = gt_x.reshape(-1,1)
                
                # hyp datas
                illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
                I = I.reshape(-1, 1)
                hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                hyp = hyp.reshape(-1, 1)
                
                # hyp gt data
                shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1,1) # 29M, 1
                x_gt = shading_term * hyp
                
                # normalization of N3_arr
                N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
                N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
                N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
                N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)

                # model coord
                pred_xy = model(N3_arr_normalized)
                loss_depth = loss_fn(gt_x, pred_xy)
                
                # hyp lstsq
                x = compute_hyp(arg, illum, cam_crf, I, batch_size)
                loss_hyp = diff_hyp(arg, x, x_gt, batch_size)

                # loss
                losses_depth.append(loss_depth.item())
                losses_hyp.append(loss_hyp.item())
                
                # loss = (loss_depth / (1/arg.proj_H)) * arg.weight_depth
                # loss = (loss_depth / (1/arg.proj_H)) * arg.weight_depth + loss_hyp * arg.weight_hyp
                loss = (loss_depth * 10 ) * arg.weight_depth + loss_hyp * arg.weight_hyp
                losses_total.append(loss.item())
            
                total_iter += 1
                
                # model save
                if (epoch%10 == 0) or (epoch == arg.epoch_num-1):
                    if not os.path.exists(arg.model_dir):
                        os.mkdir(arg.model_dir)
                    torch.save(model.state_dict(), os.path.join(arg.model_dir, 'model_coord_%05d.pth'%epoch))
                    # torch.save(optimizer.state_dict(), os.path.join(arg.model_dir, 'optim_coord_%d.pth'%epoch))
                    
            epoch_total_loss = (sum(losses_total)/ total_iter)
            epoch_valid_depth_px = (sum(losses_depth)/ total_iter) / (1/arg.proj_H)
            epoch_valid_depth = (sum(losses_depth)/ total_iter) * 10 
            epoch_valid_hyp = (sum(losses_hyp)/total_iter)
                
            print("{%dth epoch} Valid Depth loss :"  %(epoch), epoch_valid_depth_px)
            writer.add_scalars("Valid Depth",  {'depth_loss' : epoch_valid_depth, 'weight_loss': epoch_total_loss}, epoch)
            # writer.add_scalar("Valid Depth", epoch_valid_depth_px, epoch)

            print("{%dth epoch} Valid Hyp Error: "%(epoch), epoch_valid_hyp)
            writer.add_scalars("Valid Hyp", {'hyp_loss' : epoch_valid_hyp, 'weight_loss' : epoch_total_loss}, epoch)
            
            if epoch % 30 == 0:
                
                losses_total = []
                losses_depth = []
                losses_hyp = []
                total_iter = 0
                
                
                for i, data in enumerate(eval_loader):
                    # datas
                    depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
                    print(f'rendering for {depth.shape[0]} scenes at {i}-th iteration')
                    # image formation
                    N3_arr, gt_xy, xy_real, illum_data, shading = pixel_renderer.render(depth = depth, 
                                                                normal = normal, hyp = hyp, occ = occ, 
                                                                cam_coord = cam_coord, eval = True, illum_opt=illum_opt)
                    # batch size
                    batch_size = N3_arr.shape[0]
                    pixel_num = N3_arr.shape[1]
                    
                    # to device
                    # B, # pixel, N, 3 / B, # pixel, 2 / B, # pixel, N, 29 / B, # pixel, 29 / B, 3(m), 29(wvl), # pixel
                    N3_arr, gt_x, illum_data, hyp, shading = N3_arr.to(arg.device), gt_xy[...,0].to(arg.device), illum_data.to(arg.device), hyp.to(arg.device), shading.to(arg.device)                
                    
                    # reshape
                    N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
                    I = N3_arr.reshape(-1, arg.illum_num * 3)
                
                    N3_arr = N3_arr.unsqueeze(dim = 1) 
                    gt_x = gt_x.reshape(-1,1)
                    
                    # hyp datas
                    illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
                    I = I.reshape(pixel_num, 1, 3*arg.illum_num, 1) # M, 1, 3*42, 1
                    hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                    hyp = hyp.reshape(-1, 1)          
                    
                    # hyp gt data
                    shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1,1) #29M, 1
                    x_gt = shading_term * hyp
                
                    # normalization of N3_arr
                    N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
                    N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
                    N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
                    N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)

                    # model coord
                    pred_xy = model(N3_arr_normalized) # B * # of pixel, 2
                    
                    # Nan indexing
                    check = torch.where(torch.isnan(pred_xy) == False)
                    pred_xy_loss = pred_xy[check]
                    gt_x_loss = gt_x[check]
                    
                    # hyp lstsq
                    A = illum * cam_crf
                    A = A.reshape(-1, batch_size * pixel_num, arg.wvl_num).unsqueeze(dim=1).permute(2,1,0,3) # M, 1, 3*N, wvls

                    # optimization for hyp reflec
                    batch_size_opt = 200000
                    num_iter = 2000
                    num_batches = int(np.ceil(pixel_num / batch_size_opt))
                    loss_f = torch.nn.L1Loss()
                    losses = []
                    X_np_all = torch.zeros(pixel_num, 1, arg.wvl_num, 1)
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size_opt
                        end_idx = min((batch_idx + 1)* batch_size_opt, pixel_num)
                        batch_size_ = end_idx - start_idx
                        A_batch = A[start_idx:end_idx]
                        B_batch = I[start_idx:end_idx]
                        # Evaluation Hyperspectral reflectance optimization
                        X_est = torch.randn(batch_size_, 1, arg.wvl_num, 1, requires_grad=True, device=arg.device)
                        optimizer_x = torch.optim.Adam([X_est], lr = 0.5)
                        scheduler_x = torch.optim.lr_scheduler.StepLR(optimizer_x, step_size=100, gamma = 0.5)

                        optimizer_x.zero_grad()
                        
                        for i in range(num_iter):
                            loss_x = loss_f(A_batch@X_est, B_batch)
                            loss_x.requires_grad_(True)
                            loss_x.backward()
                            losses.append(loss_x.item())
                            optimizer_x.step()
                            optimizer_x.zero_grad()
                            
                            if i % 100 == 0:
                                print(f"Batch {batch_idx + 1}/{num_batches}, Iteration {i}/{num_iter}, Loss: {loss_x.item()}, LR: {optimizer_x.param_groups[0]['lr']}")
                        scheduler_x.step()
                        
                        X_np_all[start_idx:end_idx] = X_est.detach().cpu().numpy()
                    X_np_all = X_np_all.numpy()
                    # A = A.reshape(-1, batch_size * 640*640, arg.wvl_num).permute(1,0,2)
                    
                    # list_A = list(A)
                    # block = torch.block_diag(*list_A)
                    
                    # x = torch.linalg.lstsq(block, I)
                    # x = x.solution
                    
                    diff = diff_hyp(arg, X_np_all, x_gt, arg.batch_size_eval)
                    # diff = abs(X_np_all-x_gt).sum() / (arg.wvl * arg.num_train_px_per_iter)
                    
                    # loss
                    loss_depth = loss_fn(gt_x_loss, pred_xy_loss)
                    losses_depth.append(loss_depth.item())
                    losses_hyp.append(diff)
                    
                    total_iter +=1
                    
                    # Nan 처리
                    pred_xy[torch.isnan(pred_xy)] = 0.
                          
                    # img = pred_xy.reshape(batch_size, pixel_num)
                    # img = img.reshape(batch_size, arg.cam_H, arg.cam_W)
                    
                    np.save(f"./prediction/prediction_xy_{epoch}.npy", pred_xy.detach().cpu().numpy())
                    np.save(f"./prediction/ground_truth_xy_{epoch}.npy", gt_xy.detach().cpu().numpy()) 
                    np.save(f"./prediction/ground_truth_xy_real_{epoch}.npy", xy_real.detach().cpu().numpy()) 
                
                epoch_eval_depth_px = (sum(losses_depth)/ total_iter) / (1/arg.proj_H)
                # epoch_eval_depth = (sum(losses_depth)/ total_iter) * 10 
                
                print("{%dth epoch} Evaluation loss :"  %(epoch), epoch_eval_depth_px)
                writer.add_scalar("eval_loss", epoch_eval_depth_px, epoch)
                # writer.add_image("eval predicted proj x", img[0], dataformats='HW')
                # print("{%dth epoch} Evaluation Hyp Error: "%(epoch), epoch_diff)
                
                torch.cuda.empty_cache()
    writer.flush()
    

if __name__ == "__main__":

    argument = Argument()
    arg = argument.parse()
    
    cam_crf = torch.tensor(np.load(os.path.join(arg.camera_response,'CRF_cam.npy')), device= arg.device).T

    # training
    train(arg, arg.epoch_num, cam_crf)
    