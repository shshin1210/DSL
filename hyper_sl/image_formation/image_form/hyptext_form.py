import torch
import tifffile, os
import numpy as np


class HyperText_formation():
    def __init__(self, arg):
        self.img_hyp_text_path = arg.img_hyp_text_path
        self.result_tmp_path = arg.result_tmp_path

    def hyp_form(self, illum_w_occ, i):
        """
        give texture to pattern img formation

        """
        hyp_img = self.bring_hyp_img(i)
        ill_77 = self.channel_change(illum_w_occ)

        img_hyp_txt = hyp_img * ill_77

        return img_hyp_txt
        
    def bring_hyp_img(self, i):
        """
        bring hyp spectral img numpy array

        """
        img_list = os.listdir(self.img_hyp_text_path)
        img_choice = img_list[i]
        hyp_img = np.load(os.path.join(self.img_hyp_text_path, img_choice)).astype(np.float32)
        hyp_img = torch.tensor(hyp_img) 

        return hyp_img

    def channel_change(self, illum_w_occ):
        """
        change illumination channel 3 to 77

        """
        for i in range(25):
            if i == 0 :
                ill_77 = np.append(illum_w_occ, illum_w_occ, axis=2)
            else:
                ill_77 = np.append(ill_77, illum_w_occ, axis=2)
        ill_77 = np.delete(ill_77, -1, axis = 2)
        ill_77 = torch.tensor(ill_77)

        return ill_77
    
    def array_to_tif(self, arr, i):
        """
        save array to tif file

        """
        arr = arr.detach().numpy()
        arr = np.swapaxes(arr,0,2)
        arr = np.swapaxes(arr,1,2)
        path = os.path.join(self.result_tmp_path, f'scene_{i}_hyp_form.tif')
        tifffile.imwrite(path, arr)