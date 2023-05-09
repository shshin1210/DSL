import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

from hyper_sl.mlp import mlp_depth, mlp_hyp
import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer
from hyper_sl.hyp_reconstruction import compute_hyp, diff_hyp, cal_A
from hyper_sl.depth_reconstruction import depthReconstruction
from hyper_sl.utils import data_process

from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print('cuda visible device count :',torch.cuda.device_count())
print('current device number :', torch.cuda.current_device())


def train(arg, epochs, cam_crf):
    writer = SummaryWriter(log_dir=arg.log_dir)
    
    train_dataset = dtools.pixelData(arg, train = True, eval = False, pixel_num = arg.num_train_px_per_iter)
    train_loader = DataLoader(train_dataset, batch_size = arg.batch_size_train, shuffle = True)

    test_dataset = dtools.pixelData(arg, train = False,eval = False, pixel_num = arg.num_train_px_per_iter)
    test_loader = DataLoader(test_dataset, batch_size = arg.batch_size_test, shuffle = True)

    eval_dataset = dtools.pixelData(arg, train = False,eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

    # bring model MLP
    model = mlp_depth(input_dim = arg.illum_num*3, output_dim = 2).to(device=arg.device)
    model_hyp = mlp_hyp(input_dim = arg.illum_num*3*(arg.wvl_num + 1), output_dim=arg.wvl_num, fdim = 1000).to(device=arg.device)
    
    # optimizer, schedular, loss function
    optimizer = torch.optim.Adam(list(model.parameters()), lr= arg.model_lr)
    scheduler = torch.optim.lr_scheduler.StepLR((optimizer), step_size= arg.model_step_size, gamma= arg.model_gamma)
    optimizer_hyp = torch.optim.Adam(list(model_hyp.parameters()), lr= arg.model_lr)
    scheduler_hyp = torch.optim.lr_scheduler.StepLR((optimizer_hyp), step_size= arg.model_step_size, gamma= arg.model_gamma)
    
    # illumination
    # illum_opt = torch.nn.Parameter(arg.illums)
    # illum_data = os.path.join(arg.illum_data_dir, 'illum_data.npy')
    # optimizer_illum = torch.optim.Adam([illum_opt], lr = arg.illum_lr)
    # scheduler_illum = torch.optim.lr_scheduler.StepLR((optimizer_illum), step_size= arg.illum_step_size, gamma= arg.illum_gamma)
    
    # loss ftn
    loss_fn = torch.nn.L1Loss()
    loss_fn.requires_grad_ = True
    
    loss_fn_hyp = torch.nn.L1Loss()
    loss_fn_hyp.requires_grad_ = True
    
    # rendering function
    pixel_renderer = renderer.PixelRenderer(arg = arg)
    
    # depth estimation function
    depth_reconstruction = depthReconstruction(arg = arg)
    
    # cam crf
    cam_crf = cam_crf[None,:,:].unsqueeze(dim = 2)
    
    for epoch in range(epochs):
        model.train()
        model_hyp.train()
        
        losses_total = []
        losses_depth = []
        losses_hyp = []
        total_iter = 0
        
        for i, data in enumerate(train_loader):
            # datas
            depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
            
            # image formation for # of pixels
            N3_arr, gt_xy, _, illum_data, shading  = pixel_renderer.render(depth = depth, 
                                                        normal = normal, hyp = hyp, occ = occ, 
                                                        cam_coord = cam_coord, eval = False, render_shading=True)
            
            # batch size
            batch_size = N3_arr.shape[0]
            pixel_num = N3_arr.shape[1]
                    
            # to device
            N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
            gt_xy = gt_xy.to(arg.device) # B, # pixel
            
            # DEPTH ESTIMATION
            # reshape
            N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)    
            N3_arr = N3_arr.unsqueeze(dim = 1) 
            gt_xy = gt_xy.reshape(-1,2)
            
            # normalization of N3_arr
            N3_arr_normalized = normalize.N3_normalize(N3_arr, arg.illum_num)

            # model coord
            pred_xy = model(N3_arr_normalized)
            loss_depth = loss_fn(gt_xy, pred_xy)
            
            # pred_depth = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, False)[...,2]
            
            # HYPERSPECTRAL ESTIMATION            
            # image formation with predicted depth
            N3_arr, gt_xy, _, illum_data, shading  = pixel_renderer.render(depth = depth, 
                                                        normal = normal, hyp = hyp, occ = occ, 
                                                        cam_coord = cam_coord, eval = False, render_shading = False)
            
            # to device
            N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
            illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
            hyp = hyp.to(arg.device) # B, # pixel, 25
            shading = shading.to(arg.device) # B, 3(m), 25(wvl), # pixel
            
            # hyp gt data
            hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
            shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
            gt_reflectance = shading_term * hyp
            # gt_reflectance = gt_reflectance.reshape(-1,1)/
            
            # Ax = b 에서 A
            illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
            A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
            I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim = 2)
            
            pred_reflectance = model_hyp(A, I)
            loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           

            # save last epoch training set
            if epoch == arg.epoch_num -1 :
                torch.save(N3_arr, os.path.join(arg.output_dir, f'N3_arr_{epoch}.pt'))
                torch.save(pred_xy, os.path.join(arg.output_dir, f'pred_xy_{epoch}.pt'))
                torch.save(pred_reflectance, os.path.join(arg.output_dir, f'pred_hyp_{epoch}.pt'))
            
            losses_depth.append(loss_depth.item())
            losses_hyp.append(loss_hyp.item())
            
            # loss = (loss_depth / (1/arg.proj_H)) * arg.weight_depth
            # loss = (loss_depth / (1/arg.proj_H)) * arg.weight_depth + loss_hyp * arg.weight_hyp
            loss = (loss_depth * 10) * arg.weight_depth + loss_hyp * arg.weight_hyp
            losses_total.append(loss.item())
            
            optimizer.zero_grad()            
            optimizer_hyp.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_hyp.step()
            total_iter += 1
            
        scheduler.step()
        scheduler_hyp.step()
        epoch_train_depth_px = (sum(losses_depth)/total_iter) / (1/arg.proj_H)
        epoch_total_loss = (sum(losses_total)/ total_iter)
        epoch_train_depth = (sum(losses_depth)/total_iter) * 10 
        epoch_train_hyp = (sum(losses_hyp)/total_iter)
        
        print("{%dth epoch} Train depth Loss: "%(epoch), epoch_train_depth_px)
        # writer.add_scalar("Training Depth", epoch_train_depth_px, epoch)
        writer.add_scalars("Training Depth", {'depth_loss' : epoch_train_depth, 'weight_loss': epoch_total_loss}, epoch)

        print("{%dth epoch} Train Hyp Error: "%(epoch), epoch_train_hyp)
        writer.add_scalars("Training Hyp", {'hyp_loss' : epoch_train_hyp, 'weight_loss' : epoch_total_loss}, epoch)
        torch.cuda.empty_cache()
        
        # evaluation
        model.eval()
        model_hyp.eval()
        
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
                                                            cam_coord = cam_coord, eval = False, render_shading=True)
                
                # batch size
                batch_size = N3_arr.shape[0]
                pixel_num = N3_arr.shape[1]
                
                # to device
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                gt_xy = gt_xy.to(arg.device) # B, # pixel, 2
                
                # DEPTH ESTIMATION
                # reshape
                N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)                
                N3_arr = N3_arr.unsqueeze(dim = 1) 
                gt_xy = gt_xy.reshape(-1,2)
                
                # normalization of N3_arr
                N3_arr_normalized = normalize.N3_normalize(N3_arr, arg.illum_num)
                N3_arr_normalized = N3_arr_normalized.reshape(-1, 1, arg.patch_pixel_num, arg.illum_num, 3)

                # model coordinate
                pred_xy = model(N3_arr_normalized)
                gt_xy = gt_xy.reshape(-1, 1, arg.patch_pixel_num, 2)[...,4,:].squeeze()
                loss_depth = loss_fn(gt_xy, pred_xy)
                
                pred_depth = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, False)[...,2]
                
                # HYPERSPECTRAL ESTIMATION
                N3_arr, gt_xy, _, illum_data, shading  = pixel_renderer.render(depth = pred_depth, 
                                                            normal = normal, hyp = hyp, occ = occ, 
                                                            cam_coord = cam_coord, eval = False, render_shading=False)
                # to device
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                hyp = hyp.to(arg.device) # B, # pixel, 25
                shading = shading.to(arg.device) # B, 3(m), 25(wvl), # pixel
                
                # hyp gt data
                hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
                gt_reflectance = shading_term * hyp
                
                # Ax = b 에서 A
                illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
                A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
                I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim=2)
                
                pred_reflectance = model_hyp(A, I)
                loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           

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
                    # torch.save(model.state_dict(), os.path.join(arg.model_dir, 'model_coord_%05d.pth'%epoch))
                    torch.save(model_hyp.state_dict(), os.path.join(arg.model_dir, 'model_hyp_%05d.pth'%epoch))
                                        
            epoch_valid_depth_px = (sum(losses_depth)/ total_iter) / (1/arg.proj_H)
            epoch_total_loss = (sum(losses_total)/ total_iter)
            epoch_valid_depth = (sum(losses_depth)/ total_iter) * 10 
            epoch_valid_hyp = (sum(losses_hyp)/total_iter)
                
            print("{%dth epoch} Valid loss :"  %(epoch), epoch_valid_depth_px)
            # writer.add_scalar("Valid Depth", epoch_valid_depth, epoch)
            writer.add_scalars("Valid Depth",  {'depth_loss' : epoch_valid_depth, 'weight_loss': epoch_total_loss}, epoch)

            print("{%dth epoch} Valid Hyp Error: "%(epoch), epoch_valid_hyp)
            writer.add_scalars("Valid Hyp", {'hyp_loss' : epoch_valid_hyp, 'weight_loss' : epoch_total_loss}, epoch)
            torch.cuda.empty_cache()
            
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
                                                                cam_coord = cam_coord, eval = True, render_shading=True)
                    # batch size
                    batch_size = N3_arr.shape[0]
                    pixel_num = N3_arr.shape[1]
                    
                    # to device
                    N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                    gt_xy = gt_xy.to(arg.device) # B, # pixel, 2
                
                    # DEPTH ESTIMATION
                    gt_xy = gt_xy.reshape(-1,2)
                    
                    # N3_arr padding
                    N3_arr = data_process.to_patch(arg, N3_arr, batch_size)
                    
                    # normalization of N3_arr
                    N3_arr_normalized = normalize.N3_normalize(N3_arr, arg.illum_num)
                    N3_arr_normalized = N3_arr_normalized.reshape(-1, 1, arg.patch_pixel_num, arg.illum_num, 3)

                    # model coord
                    pred_xy = model(N3_arr_normalized) # B * # of pixel, 2
                    
                    pred_depth = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, True)[...,2]

                    # Nan indexing
                    check = torch.where(torch.isnan(pred_xy) == False)
                    pred_xy_loss = pred_xy[check]
                    gt_xy_loss = gt_xy[check]
                    loss_depth = loss_fn(gt_xy_loss, pred_xy_loss)
                    
                    # HYPERSPECTRAL ESTIMATION
                    N3_arr, gt_xy, _, illum_data, shading  = pixel_renderer.render(depth = pred_depth, 
                                                                normal = normal, hyp = hyp, occ = occ, 
                                                                cam_coord = cam_coord, eval = False, render_shading=False)
                    
                    # to device
                    N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                    illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                    hyp = hyp.to(arg.device) # B, # pixel, 25
                    shading = shading.to(arg.device) # B, 3(m), 25(wvl), # pixel
                    
                    # hyp gt data
                    hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                    shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
                    gt_reflectance = shading_term * hyp
                    
                    # Ax = b 에서 A
                    illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29
                    A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
                    I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim = 2)

                    pred_reflectance = model_hyp(A, I)
                    loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           
            
                    # loss
                    losses_depth.append(loss_depth.item())
                    losses_hyp.append(loss_hyp.item())

                    loss = (loss_depth * 10) * arg.weight_depth + loss_hyp * arg.weight_hyp
                    losses_total.append(loss.item())
            
                    total_iter +=1
                    
                    # Nan 처리
                    pred_xy[torch.isnan(pred_xy)] = 0.
                    
                    np.save(f"./prediction/prediction_xy_{epoch}.npy", pred_xy.detach().cpu().numpy())
                    np.save(f"./prediction/ground_truth_xy_{epoch}.npy", gt_xy.detach().cpu().numpy()) 
                    np.save(f"./prediction/ground_truth_hyp_{epoch}.npy", gt_reflectance.detach().cpu().numpy()) 
                    np.save(f"./prediction/prediction_hyp_{epoch}.npy", pred_reflectance.detach().cpu().numpy()) 

                epoch_eval_depth_px = (sum(losses_depth)/ total_iter) / (1/arg.proj_H)
                epoch_total_loss = (sum(losses_total)/ total_iter)
                epoch_eval_depth = (sum(losses_depth)/ total_iter) * 10 
                epoch_eval_hyp = (sum(losses_hyp)/total_iter)
                
                print("{%dth epoch} Eval loss :"  %(epoch), epoch_eval_depth_px)
                # writer.add_scalar("eval_loss", epoch_eval_depth_px, epoch)
                writer.add_scalars("Eval Depth",  {'depth_loss' : epoch_eval_depth, 'weight_loss': epoch_total_loss}, epoch)
                
                print("{%dth epoch} Eval Hyp Error: "%(epoch), epoch_eval_hyp)
                writer.add_scalars("Eval Hyp", {'hyp_loss' : epoch_eval_hyp, 'weight_loss' : epoch_total_loss}, epoch)
                torch.cuda.empty_cache()
    writer.flush()
    

if __name__ == "__main__":

    argument = Argument()
    arg = argument.parse()
    
    from hyper_sl.utils import normalize
    from hyper_sl.image_formation import camera
    
    cam_crf = camera.Camera(arg).get_CRF()
    cam_crf = torch.tensor(cam_crf, device= arg.device).T

    # training
    train(arg, arg.epoch_num, cam_crf)
    