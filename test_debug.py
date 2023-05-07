import torch
from torch.utils.data import DataLoader
import numpy as np

import os, cv2
from hyper_sl.utils.ArgParser import Argument

from hyper_sl.mlp import mlp_depth, mlp_hyp
import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer
from hyper_sl.hyp_reconstruction import cal_A
from hyper_sl.depth_reconstruction import depthReconstruction
from hyper_sl.utils import data_process

from scipy.io import loadmat

from hyper_sl.data import create_data_patch
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('cuda visible device count :',torch.cuda.device_count())
print('current device number :', torch.cuda.current_device())


def test(arg, cam_crf, model_path, model_num):

    # bring model MLP
    model = mlp_depth(input_dim = arg.patch_pixel_num * arg.illum_num*3, output_dim = 2).to(device=arg.device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model_depth_newcal_%05d.pth' %model_num), map_location=arg.device))
    
    model_hyp = mlp_hyp(input_dim = arg.illum_num*3*(arg.wvl_num + 1), output_dim=arg.wvl_num, fdim = 1000).to(device=arg.device)
    model_hyp.load_state_dict(torch.load(os.path.join(model_path, 'model_hyp_%05d.pth' %model_num), map_location=arg.device))

    # loss ftn
    loss_fn = torch.nn.L1Loss()
    loss_fn.requires_grad_ = False
    
    loss_fn_hyp = torch.nn.L1Loss()
    loss_fn_hyp.requires_grad_ = False
    
    # rendering function
    pixel_renderer = renderer.PixelRenderer(arg = arg)
    
    # depth estimation function
    depth_reconstruction = depthReconstruction(arg = arg)
    
    # cam crf
    cam_crf = cam_crf[None,:,:].unsqueeze(dim = 2)
    
    if arg.real_data_scene:
        eval_dataset = dtools.pixelData(arg, train = True, eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False, real = True)
    else:
        eval_dataset = dtools.pixelData(arg, train = False,eval = True, pixel_num = arg.cam_H* arg.cam_H, random = False)
    
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)

    model.eval()
    model_hyp.eval()
    
    if arg.real_data_scene:
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                # datas
                N3_arr, illum_data, cam_coord = data[0], data[1], data[2]
                
                # intensity check
                N3_arr = N3_arr.reshape(580,890,40,3)
                img_intensity_g = N3_arr[200,350,:,1]

                # intensity check
                illum = np.zeros(shape = (360, 640, 40, 3))
                for i in range(arg.illum_num):
                    illum[:,:,i] = cv2.imread("./dataset/image_formation/illum/graycode_pattern/pattern_%02d.png"%i)/255.
                illum_intensity_g = illum[116, 221,:,1]

                # batch size
                batch_size = N3_arr.shape[0]
                pixel_num = N3_arr.shape[1]
                
                # to device
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3

                vis(N3_arr.reshape(580,890,40,3).detach().cpu().numpy())
                
                # DEPTH ESTIMATION
                N3_arr = N3_arr.reshape(-1,arg.illum_num, 3).unsqueeze(dim = 1)          
                
                # N3_arr padding
                N3_arr_patch = data_process.to_patch(arg, N3_arr)
                
                # normalization of N3_arr
                N3_arr_normalized = normalize.N3_normalize(N3_arr_patch, arg.illum_num)
                N3_arr_normalized = N3_arr_normalized.reshape(-1, 1, arg.patch_pixel_num, arg.illum_num, 3)
                
                # model coord
                pred_xy = model(N3_arr_normalized) # B * # of pixel, 2                    
                pred_XYZ = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, True)
                pred_depth = pred_XYZ[...,2].detach().cpu()
                
                pred_depth = np.load("./calibration/gray_code_depth_estimation.npy")
                pred_depth = torch.tensor(pred_depth[...,2]).reshape(arg.cam_H*arg.cam_W).unsqueeze(dim = 0).type(torch.float32) #.to(arg.device)
                
                print('depth_pred_finished')
                
                # HYPERSPECTRAL ESTIMATION                                   
                _, xy_proj_real_norm, illum_data, _ = pixel_renderer.render(pred_depth, None, None, None, cam_coord, None, None, True)
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                                
                illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                
                # Ax = b 에서 A
                illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29
                A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
                I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim = 2)

                # Using Network
                pred_reflectance = model_hyp(A, I)
                illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                
                # Using optimization
                # hyp gt data
                C, R, N, W, M = arg.cam_W, arg.cam_H, arg.illum_num, arg.wvl_num, arg.cam_W* arg.cam_H
                
                # A, b
                A = A.reshape(arg.cam_H, arg.cam_W, 1, arg.illum_num*3, arg.wvl_num).detach().cpu().numpy()
                b = I.reshape(arg.cam_H, arg.cam_W, arg.illum_num*3, 1).detach().cpu().numpy()
                
                # spectralon의 hyperspectral reflectance
                create_data = create_data_patch.createData
                
                pixel_num = arg.cam_H * arg.cam_W
                random = False
                index = 0
                
                hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                hyp = torch.ones_like(hyp)
                hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                # shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
                # gt_reflectance = shading_term * hyp
                
                
                # A, b into patches
                b = b.flatten()
                idx = np.where(b > 0.95)

                b[idx] = 0

                A = A.reshape(-1, W)
                A[idx] = 0

                A = A.reshape(R, C, 1, 3*N, W)
                b = b.reshape(R, C, 1, 3*N, 1)
                
                A1 = A[:290,:445]
                A2 = A[:290,445:]
                A3 = A[290:,:445]
                A4 = A[290:,445:]
                
                b1 = b[:290,:445]
                b2 = b[:290,445:]
                b3 = b[290:,:445]
                b4 = b[290:,445:]
                
                A_list = [A1,A2,A3,A4]
                b_list = [b1,b2,b3,b4]
    
                batch_size = 200000
                num_iter = 5000
                num_batches = int(np.ceil(M / batch_size))
                loss_f = torch.nn.L1Loss()
                loss_f.requires_grad_ = True
                losses = []
                X_np_all = torch.zeros(R, C, 1, W, 1)

                # define initial learning rate and decay step
                lr = 1
                decay_step = 500


                # A = A.reshape(C, R, 1, 3*N, W)
                # b = b.reshape(C, R, 1, 3*N, 1)

                r, c = 290, 445
                
                # training loop over batches
                for batch_idx in range(4):
                    # start_idx = batch_idx * batch_size
                    # end_idx = min((batch_idx + 1) * batch_size, M)
                    # batch_size_ = end_idx - start_idx
                    A_batch = torch.from_numpy(A_list[batch_idx]).to(arg.device).reshape(r*c,1, 3*N, W)
                    B_batch = torch.from_numpy(b_list[batch_idx]).to(arg.device).reshape(r*c,1, 3*N, 1)
                    X_est = torch.randn(r*c, 1, W, 1, requires_grad=True, device=arg.device)
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
                            print(f"Batch {batch_idx + 1}/{num_batches}, Iteration {i}/{num_iter}, Loss: {loss.item()}, TV Loss: {loss_tv.item()}, Spec Loss: {loss_spec.item()},  LR: {optimizer.param_groups[0]['lr']}")

                    # X_np_all[start_idx:end_idx] = X_est.detach().cpu()
                    
                    if batch_idx == 0:
                        X_np_all[:290,:445]= X_est.detach().cpu().reshape(r,c,1,W,1)
                    elif batch_idx == 1:
                        X_np_all[:290,445:]= X_est.detach().cpu().reshape(r,c,1,W,1)
                    elif batch_idx == 2:
                        X_np_all[290:,:445]= X_est.detach().cpu().reshape(r,c,1,W,1)
                    else:
                        X_np_all[290:,445:]= X_est.detach().cpu().reshape(r,c,1,W,1)

                X_np_all = X_np_all.numpy()

                
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

                X_np_all = X_np_all.reshape(C,R,W)
            
                max_images_per_column = 5
                num_columns = (W + max_images_per_column - 1) // max_images_per_column
                plt.figure(figsize=(15, 3*num_columns))

                for c in range(num_columns):
                    start_index = c * max_images_per_column
                    end_index = min(start_index + max_images_per_column, W)
                    num_images = end_index - start_index
                    
                    for i in range(num_images):
                        plt.subplot(num_columns, num_images, i + c * num_images + 1)
                        plt.imshow(X_np_all[:, :, i + start_index], vmin=0, vmax=1)
                        plt.axis('off')
                        plt.title(f"Image {i + start_index}")
                        
                        if i + start_index == W - 1:
                            plt.colorbar()
            
                plt.show()
        
    else:
        with torch.no_grad():
                
            losses_depth = []
            losses_hyp = []
            total_iter = 0
            
            for i, data in enumerate(eval_loader):
                # datas
                depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
    
                create_data = create_data_patch.createData
                
                pixel_num = arg.cam_H * arg.cam_W
                random = False
                index = 0

                # depth = create_data(arg, "depth", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                depth = torch.tensor(np.load("./calibration/spectralon_depth.npy")[...,2].reshape(1,-1)).type(torch.float32)

                # depth_linespace = torch.linspace(0.6, 0.8, 890)
                # depth_repeat = depth_linespace.repeat(580,1)
                # depth[0] = depth_repeat

                # # depth[:] = plane_XYZ.reshape(-1,3)[:,2].unsqueeze(dim =0)*1e-3
                # depth = depth.reshape(-1, 580*890)
                
                # depth = np.load("./calibration/gray_code_depth_estimation.npy")
                # depth = torch.tensor(depth[...,2]).reshape(arg.cam_H*arg.cam_W).unsqueeze(dim = 0).type(torch.float32) #.to(arg.device)
                # depth = create_data(arg, "depth", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                
                normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                normal = torch.zeros_like(normal)
                normal[:,2] = -1.
                
                hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                hyp = torch.ones_like(hyp)
                hyp[:] = 0.9
                
                occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                occ = torch.ones_like(occ)
                
                cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0)
    
                
                print(f'rendering for {depth.shape[0]} scenes at {i}-th iteration')
                # image formation
                N3_arr, gt_xy, illum_data, shading = pixel_renderer.render(depth = depth, 
                                                            normal = normal, hyp = hyp, occ = occ, 
                                                            cam_coord = cam_coord, eval = True)
                # batch size
                batch_size = N3_arr.shape[0]
                pixel_num = N3_arr.shape[1]
                
                # to device
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                
                vis(N3_arr.reshape(580,890,40,3).detach().cpu().numpy())
                
                gt_xy = gt_xy.to(arg.device) # B, # pixel, 2
            
                # DEPTH ESTIMATION
                N3_arr = N3_arr.reshape(-1,arg.illum_num, 3).unsqueeze(dim = 1)           
                gt_xy = gt_xy.reshape(-1,2)
                
                # intensity check
                illum = np.zeros(shape = (360, 640, 40, 3))
                for i in range(arg.illum_num):
                    illum[:,:,i] = cv2.imread("./dataset/image_formation/illum/graycode_pattern/pattern_%02d.png"%i)/255.

                # N3_arr padding
                N3_arr_patch = data_process.to_patch(arg, N3_arr)
                
                # normalization of N3_arr
                N3_arr_normalized = normalize.N3_normalize(N3_arr_patch, arg.illum_num)
                N3_arr_normalized = N3_arr_normalized.reshape(-1, 1, arg.patch_pixel_num, arg.illum_num, 3)
                
                # model coord
                pred_xy = model(N3_arr_normalized) # B * # of pixel, 2                    
                pred_depth = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, True)[...,2].detach().cpu()

                # Nan indexing
                check = torch.where(torch.isnan(pred_xy) == False)
                pred_xy_loss = pred_xy[check]
                gt_xy_loss = gt_xy[check]
                loss_depth = loss_fn(gt_xy_loss, pred_xy_loss)
                
                # nan 처리하기
                pred_depth[torch.isnan(pred_depth) == True] = 0.
                
                # HYPERSPECTRAL ESTIMATION                    
                N3_arr, gt_xy, illum_data, shading  = pixel_renderer.render(depth = depth, 
                                                            normal = normal, hyp = hyp, occ = occ, 
                                                            cam_coord = cam_coord, eval = False)

                
                # to device
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                hyp = hyp.to(arg.device) # B, # pixel, 25
                shading = shading.to(arg.device) # B, 3(m), 25(wvl), # pixel
                
                # hyp gt data
                occ = occ.reshape(-1,1).to(arg.device)
                hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
                gt_reflectance = shading_term * hyp * occ
                
                # Ax = b 에서 A
                illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29
                A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
                I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim = 2)

                pred_reflectance = model_hyp(A, I)
                loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           

                # loss
                losses_depth.append(loss_depth.item())
                losses_hyp.append(loss_hyp.item()* 10)

                total_iter +=1
                
                # Nan 처리
                pred_xy[torch.isnan(pred_xy)] = 0.
                
                np.save(f"./prediction/prediction_xy.npy", pred_xy.detach().cpu().numpy())
                np.save(f"./prediction/ground_truth_xy.npy", gt_xy.detach().cpu().numpy()) 
                np.save(f"./prediction/ground_truth_hyp.npy", gt_reflectance.detach().cpu().numpy()) 
                np.save(f"./prediction/prediction_hyp.npy", pred_reflectance.detach().cpu().numpy()) 

            epoch_eval_depth_px = (sum(losses_depth)/ total_iter) / (1/arg.proj_H)
            epoch_eval_hyp = (sum(losses_hyp)/total_iter)
            
            print("Eval loss :" , epoch_eval_depth_px)
            print("Eval Hyp Error: ", epoch_eval_hyp)
            
            torch.cuda.empty_cache()

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


def vis(data):
    illum_num = 40
    max_images_per_column = 5
    num_columns = (illum_num + max_images_per_column - 1) // max_images_per_column
    plt.figure(figsize=(10, 3*num_columns))

    for c in range(num_columns):
        start_index = c * max_images_per_column
        end_index = min(start_index + max_images_per_column, illum_num)
        num_images = end_index - start_index
                
        for i in range(num_images):
            plt.subplot(num_columns, num_images, i + c * num_images + 1)
            plt.imshow(data[:, :, i + start_index], vmin=0., vmax=1.)
            plt.axis('off')
            plt.title(f"Image {i + start_index}")
            cv2.imwrite('./simulated_imgs/spectralon_simulation_%04d_img.png'%(i+start_index), data[:, :, i + start_index, ::-1]*255.)
                    
            if i + start_index == illum_num - 1:
                plt.colorbar()

    plt.show()
    
if __name__ == "__main__":

    argument = Argument()
    arg = argument.parse()
    
    from hyper_sl.utils import normalize
    from hyper_sl.image_formation import camera
    
    cam_crf = camera.Camera(arg).get_CRF()
    cam_crf = torch.tensor(cam_crf, device= arg.device).T

    model_dir = arg.model_dir
    # training
    test(arg, cam_crf, model_dir, 1999)
    