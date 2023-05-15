import torch
from torch.utils.data import DataLoader
import numpy as np

import os, cv2
from hyper_sl.utils.ArgParser import Argument
from hyper_sl.data import create_data_patch

from hyper_sl.mlp import mlp_depth, mlp_hyp
import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer
from hyper_sl.hyp_reconstruction import cal_A
from hyper_sl.depth_reconstruction import depthReconstruction
from hyper_sl.utils import data_process

from scipy.io import loadmat
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('cuda visible device count :',torch.cuda.device_count())
print('current device number :', torch.cuda.current_device())


def test(arg, cam_crf, model_path, model_num):

    # bring model MLP
    # model = mlp_depth(input_dim = arg.patch_pixel_num * arg.illum_num*3, output_dim = 2).to(device=arg.device)
    # model.load_state_dict(torch.load(os.path.join(model_path, 'model_depth_newcal_%05d.pth' %model_num), map_location=arg.device))
    
    # model_hyp = mlp_hyp(input_dim = arg.illum_num*3*(arg.wvl_num + 1), output_dim=arg.wvl_num, fdim = 1000).to(device=arg.device)
    # model_hyp.load_state_dict(torch.load(os.path.join(model_path, './model_hyp_0506_line_%05d.pth' %940), map_location=arg.device))

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

    # model.eval()
    # model_hyp.eval()
    
    if arg.real_data_scene:
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                # datas
                N3_arr, illum_data, cam_coord = data[0], data[1], data[2]
                
                # intensity check
                N3_arr = N3_arr.reshape(arg.cam_H,arg.cam_W,arg.illum_num,3)

                # batch size
                batch_size = N3_arr.shape[0]
                pixel_num = N3_arr.shape[1]
                
                # to device
                N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                N3_arr = torch.clamp(N3_arr, 0, 1)
                
                # # DEPTH ESTIMATION
                # N3_arr = N3_arr.reshape(-1,arg.illum_num, 3).unsqueeze(dim = 1)          
                
                # # N3_arr padding
                # N3_arr_patch = data_process.to_patch(arg, N3_arr)
                
                # # normalization of N3_arr
                # N3_arr_normalized = normalize.N3_normalize(N3_arr_patch, arg.illum_num)
                # N3_arr_normalized = N3_arr_normalized.reshape(-1, 1, arg.patch_pixel_num, arg.illum_num, 3)
                
                # # model coord
                # pred_xy = model(N3_arr_normalized) # B * # of pixel, 2                    
                # pred_XYZ = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, True)
                # pred_depth = pred_XYZ[...,2].detach().cpu()
                
                # HYPERSPECTRAL ESTIMATION                    
                # to device         
                # depth = torch.tensor(np.load("./calibration/color_checker_depth_0508.npy")[...,2].reshape(1,-1)).type(torch.float32)
                depth = torch.tensor(np.load("./spectralon_depth_0511.npy")[...,2].reshape(1,-1)).type(torch.float32)
       
                _, xy_proj_real_norm, illum_data, _ = pixel_renderer.render(depth, None, None, None, cam_coord, None, True)
                                
                illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                
                # Ax = b 에서 A
                illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29
                A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
                I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim = 2)

                pred_reflectance = model_hyp(A, I)
                illum_data = illum_data.to(arg.device) # B, # pixel, N, 25
                
                print('end')
            
        
    else:
        with torch.no_grad():
                
            losses_depth = []
            losses_hyp = []
            total_iter = 0
            
            for i, data in enumerate(eval_loader):
                # datas
                depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
                
                # print(f'rendering for {depth.shape[0]} scenes at {i}-th iteration')
                # # image formation
                # N3_arr, gt_xy, illum_data, shading = pixel_renderer.render(depth = depth, 
                #                                             normal = normal, hyp = hyp, occ = occ, 
                #                                             cam_coord = cam_coord, eval = True)
                # # batch size
                # batch_size = N3_arr.shape[0]
                # pixel_num = N3_arr.shape[1]
                
                # # to device
                # N3_arr = N3_arr.to(arg.device) # B, # pixel, N, 3
                
                # vis(N3_arr.reshape(580,890,40,3).detach().cpu().numpy())
                
                # gt_xy = gt_xy.to(arg.device) # B, # pixel, 2
            
                # # DEPTH ESTIMATION
                # N3_arr = N3_arr.reshape(-1,arg.illum_num, 3).unsqueeze(dim = 1)           
                # gt_xy = gt_xy.reshape(-1,2)

                # # N3_arr padding
                # N3_arr = data_process.to_patch(arg, N3_arr)

                # # normalization of N3_arr
                # N3_arr_normalized = normalize.N3_normalize(N3_arr, arg.illum_num)
                # N3_arr_normalized = N3_arr_normalized.reshape(-1, 1, arg.patch_pixel_num, arg.illum_num, 3)
                
                # # model coord
                # pred_xy = model(N3_arr_normalized) # B * # of pixel, 2                    
                # pred_depth = depth_reconstruction.depth_reconstruction(pred_xy, cam_coord, True)[...,2].detach().cpu()

                # # Nan indexing
                # check = torch.where(torch.isnan(pred_xy) == False)
                # pred_xy_loss = pred_xy[check]
                # gt_xy_loss = gt_xy[check]
                # loss_depth = loss_fn(gt_xy_loss, pred_xy_loss)
                
                # # nan 처리하기
                # pred_depth[torch.isnan(pred_depth) == True] = 0.
                
                create_data = create_data_patch.createData
                
                pixel_num = arg.cam_H * arg.cam_W
                random = False
                index = 0

                # depth = torch.tensor(np.load("./spectralon_depth_0510.npy")[...,2].reshape(1,-1)).type(torch.float32)
                depth = torch.tensor(np.load("./calibration/spectralon_depth_0511.npy")[...,2].reshape(1,-1)).type(torch.float32).to(arg.device)
                # depth_linespace = torch.linspace(0.9, 0.95, 580)
                # depth_repeat = depth_linespace.repeat(890,1).T
                # depth[0] = depth_repeat.reshape(1,-1)
                # depth[0,:] = 0.95
                                
                normal = create_data(arg, "normal", pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                normal = torch.zeros_like(normal)
                normal[:,2] = -1.
                
                hyp = create_data(arg, 'hyp', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                hyp = torch.ones_like(hyp)
                hyp[:] = 0.9
                
                occ = create_data(arg, 'occ', pixel_num, random = random, i = index).create().unsqueeze(dim = 0)
                occ = torch.ones_like(occ)
                
                cam_coord = create_data(arg, 'coord', pixel_num, random = random).create().unsqueeze(dim = 0).to(arg.device)
    
                # HYPERSPECTRAL ESTIMATION                    
                N3_arr, gt_xy, illum_data, shading  = pixel_renderer.render(depth = depth, 
                                                            normal = normal, hyp = hyp, occ = occ, 
                                                            cam_coord = cam_coord, eval = False)

                batch_size = N3_arr.shape[0]
                pixel_num = N3_arr.shape[1]
                
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
                # vis(pred_reflectance.reshape(580,890,25).detach().cpu().numpy(), 25, 0., 0.1)
                print('visualization')
                loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           

                # loss
                # losses_depth.append(loss_depth.item())
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
    
def vis(data, num, vmin, vmax):
    illum_num = num
    max_images_per_column = 5
    num_columns = (illum_num + max_images_per_column - 1) // max_images_per_column
    plt.figure(figsize=(10, 3*num_columns))

    for c in range(num_columns):
        start_index = c * max_images_per_column
        end_index = min(start_index + max_images_per_column, illum_num)
        num_images = end_index - start_index
                
        for i in range(num_images):
            plt.subplot(num_columns, num_images, i + c * num_images + 1)
            plt.imshow(data[:, :, i + start_index], vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.title(f"Image {i + start_index}")
            cv2.imwrite('./spectralon/spectralon_real_%04d_img.png'%(i+start_index), data[:, :, i + start_index, ::-1]*255.)
                    
            if i + start_index == illum_num - 1:
                plt.colorbar()

    # plt.show()
    
    
if __name__ == "__main__":

    argument = Argument()
    arg = argument.parse()
    
    from hyper_sl.utils import normalize
    from hyper_sl.image_formation import camera
    
    cam_crf = camera.Camera(arg).get_CRF()
    cam_crf = torch.tensor(cam_crf, device= arg.device).T

    # date = "0505"
    # argumetns = pathname = 'gamma%03f_illum%03f_noise%03f_normal'%(0.8, 0.5, 0.015)
    # model_dir = os.path.join(arg.model_dir, date, argumetns)
    
    
    model_dir = "/log/hyp-3d-imaging/result/model_graycode/0505/my_local"
    
    # model_dir = "/log/hyp-3d-imaging/result/model_hyp_0504_previous"
    # training
    test(arg, cam_crf, model_dir, 1999)
    