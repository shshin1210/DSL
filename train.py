import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from hyper_sl.utils.ArgParser import Argument

from hyper_sl.mlp import MLP
import hyper_sl.datatools as dtools 

from torch.utils.tensorboard import SummaryWriter

print('torch cuda available : ', torch.cuda.is_available())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(arg, epochs, cam_crf):
    
    writer = SummaryWriter(log_dir=arg.log_dir)
    
    train_dataset = dtools.hypData(arg, train = True, eval = False,pixel_num= arg.num_train_px_per_iter, bring_data = arg.load_dataset)
    train_loader = DataLoader(train_dataset, batch_size = arg.batch_size_train, shuffle = True)

    test_dataset = dtools.hypData(arg, train = False,eval = False, pixel_num= arg.num_train_px_per_iter, bring_data = arg.load_dataset)
    test_loader = DataLoader(test_dataset, batch_size = arg.batch_size_test, shuffle = True)

    eval_dataset = dtools.hypData(arg, train = False,eval = True, pixel_num= arg.cam_H*arg.cam_W, bring_data = arg.load_dataset, random = False)
    eval_loader = DataLoader(eval_dataset, batch_size= arg.batch_size_eval, shuffle=True)
    
    # bring model MLP
    model = MLP(input_dim = arg.illum_num*3, output_dim = 1).to(device=arg.device) 
    
    # optimizer, schedular, loss function
    optimizer = torch.optim.Adam(list(model.parameters()), lr= 5*1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR((optimizer), step_size=350, gamma=0.9)

    loss_fn = torch.nn.L1Loss()
    loss_fn.requires_grad_ = True
    
    cam_crf = cam_crf.to(arg.device)
    
    for epoch in range(epochs):
        model.train()

        losses = []
        total_iter = 0
        
        for i, data in enumerate(train_loader):
            N3_arr = data[0].to(arg.device) # B, # pixel, N, 3
            gt_xy = data[1].to(arg.device) # B, # pixel, 2
            illum = data[2].to(arg.device) # B, # pixel, 42, 29 / illumination
            hyp_gt = data[3].to(arg.device)
            
            # reshape
            N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
            N3_arr = N3_arr.unsqueeze(dim = 1)
            # gt_xy = gt_xy.reshape(-1,2)
            gt_xy = gt_xy[...,0]
            gt_xy = gt_xy.reshape(-1,1)
            illum = illum.reshape(-1, arg.illum_num, arg.wvl_num)
            illum = illum.unsqueeze(dim = 1) # B * # pixel, 1, 42, 29
            hyp_gt = hyp_gt.reshape(-1, arg.wvl_num)
            
            # normalization of N3_arr
            N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
            N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
            N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
            N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)

            # model coord
            pred_xy = model(N3_arr_normalized)
            loss = loss_fn(gt_xy, pred_xy)
            
            # save last epoch training set
            if epoch == arg.epoch_num -1 :
                torch.save(N3_arr, os.path.join(arg.output_dir, f'N3_arr_{epoch}.pt'))
                torch.save(pred_xy, os.path.join(arg.output_dir, f'pred_xy_{epoch}.pt'))
            
            losses.append(loss.item())
            
            # optmizer zero grad
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            total_iter += 1
        
        # scheduler
        scheduler.step()
        
        epoch_train_loss = (sum(losses)/total_iter)/ (1/arg.proj_H)
        
        print("{%dth epoch} Train Loss: "%(epoch), epoch_train_loss)
        writer.add_scalar("train_loss", epoch_train_loss, epoch)

        # evaluation
        model.eval()

        
        with torch.no_grad():
            valid_losses = []
            total_iter = 0
            
            for i, data in enumerate(test_loader):    
                N3_arr = data[0].to(arg.device) # B, # pixel, N, 3
                gt_xy = data[1].to(arg.device) # B, # pixel, 2
                illum = data[2].to(arg.device) # B, # pixel, 42, 29 / illumination
                hyp_gt = data[3].to(arg.device)
                
                # reshape
                N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
                N3_arr = N3_arr.unsqueeze(dim = 1)
                # gt_xy = gt_xy.reshape(-1,2)
                gt_xy = gt_xy[...,0]
                gt_xy = gt_xy.reshape(-1,1)
                illum = illum.reshape(-1, arg.illum_num, arg.wvl_num)
                illum = illum.unsqueeze(dim = 1)
                hyp_gt = hyp_gt.reshape(-1, arg.wvl_num)

                # normalization of N3_arr
                N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
                N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
                N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
                N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)
                
                # model coord
                pred_xy = model(N3_arr_normalized) # B * # of pixel, 2
                loss = loss_fn(gt_xy, pred_xy)

                valid_losses.append(loss.item())
                
                total_iter += 1
                
            # model save
            if (epoch%10 == 0) or (epoch == arg.epoch_num-1):
                if not os.path.exists(arg.model_dir):
                    os.mkdir(arg.model_dir)
                torch.save(model.state_dict(), os.path.join(arg.model_dir, 'model_coord_%d.pth'%epoch))
            
            epoch_valid_loss = (sum(valid_losses)/ total_iter) / (1/arg.proj_H)
                
            print("{%dth epoch} Valid loss :"  %(epoch), epoch_valid_loss)
            writer.add_scalar("valid_loss", epoch_valid_loss, epoch)

            if epoch % 30 == 0:
                eval_losses = []
                
                total_iter = 0
                for i, data in enumerate(eval_loader):
                    # cam coord, gt xyz return
                    N3_arr = data[0].to(arg.device) # B, # pixel, N, 3
                    gt_xy = data[1].to(arg.device) # B, # pixel, 2
                    xy_real = data[2]
                    illum = data[3].to(arg.device) # B, # pixel, 42, 29 / illumination
                    hyp_gt = data[4].to(arg.device)
                    occ_data = data[5].to(arg.device) # B, # pixel
                    
                    # reshape
                    N3_arr = N3_arr.reshape(-1,arg.illum_num, 3)
                    N3_arr = N3_arr.unsqueeze(dim = 1)
                    # gt_xy = gt_xy.reshape(-1,2)
                    gt_xy = gt_xy[...,0]
                    gt_xy = gt_xy.reshape(-1,1)
                    illum = illum.reshape(-1, arg.illum_num, arg.wvl_num)
                    illum = illum.unsqueeze(dim = 1)
                    hyp_gt = hyp_gt.reshape(-1, arg.wvl_num)
                    occ_data = occ_data.reshape(-1,1)

                    # normalization of N3_arr
                    N3_arr_r = N3_arr.reshape(-1,arg.illum_num*3)
                    N3_arr_max = N3_arr_r.max(axis = 1).values[:,None, None, None]
                    N3_arr_min = N3_arr_r.min(axis = 1).values[:,None, None, None]
                    N3_arr_normalized = (N3_arr - N3_arr_min)/(N3_arr_max - N3_arr_min)
                    
                    # model coord
                    pred_xy = model(N3_arr_normalized) # B * # of pixel, 2
                    
                                        
                    # Nan 처리
                    check = torch.where(torch.isnan(pred_xy) == False)
                    pred_xy_loss = pred_xy[check]
                    gt_xy_loss = gt_xy[check]
                    
                    loss = loss_fn(gt_xy_loss, pred_xy_loss)
                    
                    eval_losses.append(loss.item())
                    
                    total_iter +=1
    
                    # Nan 처리
                    pred_xy[torch.isnan(pred_xy)] = 0.
                    
                    np.save(f"./prediction/prediction_xy_{epoch}.npy", pred_xy.detach().cpu().numpy())
                    np.save(f"./prediction/ground_truth_xy_{epoch}.npy", gt_xy.detach().cpu().numpy()) 
                    np.save(f"./prediction/ground_truth_xy_real_{epoch}.npy", xy_real.detach().cpu().numpy()) 
                    np.save(f"./prediction/ground_truth_hyp_{epoch}.npy", hyp_gt.detach().cpu().numpy()) 
                
                epoch_eval_loss = (sum(eval_losses)/ total_iter) / (1/arg.proj_H)
                    
                print("{%dth epoch} Evaluation loss :"  %(epoch), epoch_eval_loss)
                writer.add_scalar("eval_loss", epoch_eval_loss, epoch)

    # torch.cuda.empty_cache()
    writer.flush()
    

if __name__ == "__main__":

    argument = Argument()
    arg = argument.parse()
    
    import numpy as np
    
    cam_crf = torch.tensor(np.load(os.path.join(arg.camera_response,'CRF_cam.npy')), device= arg.device)
    # training
    train(arg, arg.epoch_num, cam_crf)
    