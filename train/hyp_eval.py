import torch, cv2
from hyper_sl.utils import data_process, load_data
from hyper_sl.utils import ArgParser
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim 

class hyperspectral_evaluation():
    def __init__(self, arg):
        self.arg = arg
        
        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        self.illum_num = arg.illum_num
        self.wvl_num = arg.wvl_num
        
    def visualization(self, data, vmax):
        
        data = data.reshape(self.cam_H, self.cam_W, self.wvl_num)
        
        max_images_per_column = 5
        num_columns = (self.illum_num + max_images_per_column - 1) // max_images_per_column
        plt.figure(figsize=(15, 3*num_columns))

        for c in range(num_columns):
            start_index = c * max_images_per_column
            end_index = min(start_index + max_images_per_column, self.wvl_num)
            num_images = end_index - start_index
            
            for i in range(num_images):
                plt.subplot(num_columns, num_images, i + c * num_images + 1)
                plt.imshow(data[:, :, i + start_index], vmin=0, vmax=vmax)
                plt.axis('off')
                plt.title(f"Image {i + start_index}")
                
                if i + start_index == self.illum_num - 1:
                    plt.colorbar()
        plt.colorbar()
        plt.show()


    def hyp_evaluation(self, pred_hyp, gt_hyp, eval_type, i, eval_range = "25"):
        # reshaping
        pred_hyp = pred_hyp.reshape(self.cam_H, self.cam_W, self.wvl_num).transpose(2,0,1)
        gt_hyp = gt_hyp.reshape(self.cam_H, self.cam_W, self.wvl_num).transpose(2,0,1)
        
        occ = load_data.load_data(self.arg).load_occ(i)
        dilation_occ = data_process.dilation(occ)
        
        # occlusion
        pred_hyp = pred_hyp*dilation_occ[np.newaxis,:,:]
        gt_hyp = gt_hyp*dilation_occ[np.newaxis,:,:]

        if eval_range == "21":
            pred_hyp = pred_hyp[2:-2]
            gt_hyp = gt_hyp[2:-2]
        
        if eval_type == "psnr":
            psnr = cv2.PSNR(pred_hyp*255, gt_hyp*255)
            return psnr
        
        elif eval_type == 'sam':
            pred_hyp = pred_hyp.reshape(-1, self.cam_H*self.cam_W)
            gt_hyp = gt_hyp.reshape(-1, self.cam_H*self.cam_W)
            
            gt_norm = np.linalg.norm(gt_hyp, axis =0)
            pred_norm = np.linalg.norm(pred_hyp, axis = 0)
            
            for i in range(len(gt_norm)):
                if gt_norm[i] == 0.:
                    gt_norm[i] = 1
            for i in range(len(gt_norm)):
                if pred_norm[i] == 0.:
                    pred_norm[i] = 1
            
            gt_unit = gt_hyp / gt_norm
            pred_unit = pred_hyp / pred_norm
            
            cos = gt_unit*pred_unit
            cos_sum = cos.sum(axis=0)
            cos_mean = cos_sum.mean()
            
            theta = np.arccos(cos_mean)
            
            return theta
            
        else:
            pred_hyp = pred_hyp.reshape(-1, self.cam_H*self.cam_W)
            gt_hyp = gt_hyp.reshape(-1, self.cam_H*self.cam_W)
            
            ssim1 = ssim(pred_hyp, gt_hyp, data_range= gt_hyp.max() - gt_hyp.min())
            
            return ssim1

    def hyp_error(self, pred_hyp, gt_hyp, i):
        pred_hyp = pred_hyp.reshape(self.cam_H, self.cam_W, self.wvl_num)
        gt_hyp = gt_hyp.reshape(self.cam_H, self.cam_W, self.wvl_num)
        
        occ = load_data.load_data(self.arg).load_occ(i)
        dilation_occ = data_process.dilation(occ)
        
        error = abs(pred_hyp- gt_hyp)*dilation_occ[:,:,np.newaxis]

        return error
             
        

if __name__ == "__main__":
    parser = ArgParser.Argument()
    arg = parser.parse()
    
    pred_hyp = np.load('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/prediction/prediction_hyp_960.npy').astype(np.float32)
    gt_hyp = np.load('C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/prediction/ground_truth_hyp_960.npy').astype(np.float32)
    
    # visualization prediction
    hyperspectral_evaluation(arg).visualization(pred_hyp, 1)
    # visualization gt
    hyperspectral_evaluation(arg).visualization(gt_hyp, 1)
    
    # error
    error = hyperspectral_evaluation(arg).hyp_error(pred_hyp, gt_hyp, 0)
    
    # visualization error
    hyperspectral_evaluation(arg).visualization(error, 0.05) 
    
    # pred_hyp, gt_hyp, eval_type, i, eval_range = "25"
    psnr = hyperspectral_evaluation(arg).hyp_evaluation(pred_hyp, gt_hyp, "psnr", 0, "21")
    sam = hyperspectral_evaluation(arg).hyp_evaluation(pred_hyp, gt_hyp, "sam", 0, "21")
    ssim1 = hyperspectral_evaluation(arg).hyp_evaluation(pred_hyp, gt_hyp, "ssim", 0, "21")
    
    print(psnr, sam, ssim1)