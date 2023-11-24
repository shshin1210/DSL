import numpy as np
import torch 

depth_start = 600
depth_end = 900     
depth_arange = np.arange(depth_start, depth_end + 1, 1)
new_wvls = torch.linspace(430*1e-9, 660*1e-9, 47) # 400 ~ 680 까지 10nm 간격으로
cam_H, cam_W = 580, 890

first_illum_idx = np.load(('./dataset/image_formation/20231007/npy_data/first_illum_idx_final_transp_test_2.npy'))
first_illum_idx = first_illum_idx.reshape(len(depth_arange), len(new_wvls), cam_H*cam_W)

Mask = np.ones_like(first_illum_idx)

new_wvls = torch.linspace(430*1e-9, 660*1e-9, 47) # 400 ~ 680 까지 10nm 간격으로

for i in range(first_illum_idx.shape[0]):
    for j in range(first_illum_idx.shape[2]):
        # Get a [47,] array
        sub_array = first_illum_idx[i, :, j]
        
        # Find unique elements and their counts
        unique_elements, counts = np.unique(sub_array, return_counts=True)
        
        # Identify the repeated elements
        repeated_elements = unique_elements[counts > 1]
        
        # Find and print the indices of the repeated elements
        for element in repeated_elements:
            indices = np.where(sub_array == element)[0]
            if np.any(indices < 34):
                print('error, i, j')
            Mask[i,indices[1:],j] = 0
            
np.save('Mask.npy', Mask)