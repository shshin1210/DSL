import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

from hyper_sl.mlp import mlp_depth, mlp_hyp
import hyper_sl.datatools as dtools 
from hyper_sl.image_formation import renderer
from hyper_sl.hyp_reconstruction import cal_A

from hyper_sl.utils import data_process

from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    model_hyp = mlp_hyp(input_dim = arg.illum_num*3*(arg.wvl_num + 1), output_dim=arg.wvl_num, fdim = 1000).to(device=arg.device)
    
    # optimizer, schedular, loss function
    optimizer_hyp = torch.optim.Adam(list(model_hyp.parameters()), lr= 5*1e-4)
    scheduler_hyp = torch.optim.lr_scheduler.StepLR((optimizer_hyp), step_size= 200, gamma= 0.7)

    print("model gamma: %f, noise std: %f, illum weight: %f " %(arg.model_gamma, arg.noise_std, arg.illum_weight))

    # loss ftn   
    loss_fn_hyp = torch.nn.L1Loss()
    loss_fn_hyp.requires_grad_ = True
    
    # rendering function
    pixel_renderer = renderer.PixelRenderer(arg = arg)

    # cam crf
    cam_crf = cam_crf[None,:,:].unsqueeze(dim = 2)
    
    for epoch in range(epochs):
        model_hyp.train()

        losses_hyp = []
        total_iter = 0
        
        for i, data in enumerate(train_loader):
            # datas
            depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
            print(f'rendering for {depth.shape[0]} scenes at {i}-th iteration')
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
            occ = occ.to(arg.device).reshape(-1,1)
            
            # hyp gt data
            hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
            shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
            gt_reflectance = shading_term * hyp * occ
            
            # Ax = b 에서 A
            illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
            A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
            I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim = 2)
            
            pred_reflectance = model_hyp(A, I)
            loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           

            # save last epoch training set
            if epoch == arg.epoch_num -1 :
                torch.save(N3_arr, os.path.join(arg.output_dir, f'N3_arr_{epoch}.pt'))
            
            losses_hyp.append(loss_hyp.item() * 10)
            
            loss = loss_hyp * 10
            
            optimizer_hyp.zero_grad()
            loss.backward()
            optimizer_hyp.step()
            total_iter += 1
            
        scheduler_hyp.step()

        epoch_train_hyp = (sum(losses_hyp)/total_iter)
        
        print("{%dth epoch} Train Hyp Error: "%(epoch), epoch_train_hyp)
        writer.add_scalar('hyp_loss' , epoch_train_hyp, epoch)
        torch.cuda.empty_cache()
        
        # evaluation
        model_hyp.eval()
        
        with torch.no_grad():

            losses_hyp = []
            total_iter = 0
            
            for i, data in enumerate(test_loader):   
                # datas
                depth, normal, hyp, occ, cam_coord = data[0], data[1], data[2], data[3], data[4]
                print(f'rendering for {depth.shape[0]} scenes at {i}-th iteration')
                
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
                occ = occ.to(arg.device).reshape(-1,1)
                
                # hyp gt data
                hyp = hyp.reshape(-1, arg.wvl_num) # M, 29
                shading_term = shading[:,0,:,:].permute(0,2,1).reshape(-1, arg.wvl_num) # 29M, 1
                gt_reflectance = shading_term * hyp * occ
                
                # Ax = b 에서 A
                illum = illum_data.reshape(-1, arg.illum_num, arg.wvl_num).permute(1,0,2).unsqueeze(dim = 1) # N, 1, M, 29            
                A = cal_A(arg, illum, cam_crf, batch_size, pixel_num)
                I = N3_arr.reshape(-1, arg.illum_num * 3).unsqueeze(dim=2)

                pred_reflectance = model_hyp(A, I)
                loss_hyp = loss_fn_hyp(gt_reflectance, pred_reflectance)           

                # loss
                losses_hyp.append(loss_hyp.item() * 10 )

                loss = loss_hyp * 10 
            
                total_iter += 1
                
                # model save
                if (epoch%10 == 0) or (epoch == arg.epoch_num-1):
                    if not os.path.exists(arg.model_dir):
                        os.mkdir(arg.model_dir)
                    torch.save(model_hyp.state_dict(), os.path.join(arg.model_dir, 'model_hyp_0506_line_%05d.pth'%epoch))
                                        
            epoch_valid_hyp = (sum(losses_hyp)/total_iter)
         

            print("{%dth epoch} Valid Hyp Error: "%(epoch), epoch_valid_hyp)
            writer.add_scalar("Valid Hyp", epoch_valid_hyp, epoch)
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
    