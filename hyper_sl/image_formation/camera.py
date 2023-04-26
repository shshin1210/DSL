import torch
import numpy as np
import os
from scipy.interpolate import interp1d

class Camera():
    def __init__(self, arg):
        
        self.crf_dir = arg.camera_response
        
    def intrinsic_cam(self):        
        intrinsic_cam = torch.tensor([[1.7471120984549243e+03, 0.00000000e+00, 4.3552404635908243e+02],
                                      [0.00000000e+00 ,1.7562111249245049e+03 ,3.3663669106446793e+02] ,
                                      [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
            
        return intrinsic_cam
    
    #### Unprojection
    def unprojection(self, depth, cam_coord):
        """ Unproject camera sensor coord plane to world coord

            input : depth
            return : world coordinate X,Y,Z
        """
        
        suv1 = cam_coord*depth.unsqueeze(dim = 2)
        XYZ = torch.linalg.inv(self.intrinsic_cam())@(suv1.permute(0,2,1))
        
        X, Y, Z = XYZ[:,0,:], XYZ[:,1,:], XYZ[:,2,:]
        return X, Y, Z
        

    def get_CRF(self):
        # CRF = self.compute_CRF()
        CRF = np.load(os.path.join(self.crf_dir, 'crf_cam_new_v2.npy'))
        map_scale = interp1d([CRF.min(), CRF.max()], [0.,1.])
        CRF = torch.tensor(map_scale(CRF).astype(np.float32))        
        CRF = CRF[2:27]
        return CRF
        
        