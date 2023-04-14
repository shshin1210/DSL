import torch
import numpy as np
import os

class Camera():
    def __init__(self, arg):
        
        self.crf_dir = arg.camera_response
        
    def intrinsic_cam(self):
        # intrinsic_cam = torch.tensor([[1.73451929e+03, 0.00000000e+00 ,4.44392360e+02],
        #                             [0.00000000e+00 ,1.71309364e+03 , 4.18639134e+02],
        #                             [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
        
        intrinsic_cam = torch.tensor([[1.73445592e+03, 0.00000000e+00, 3.69434796e+02],
                                      [0.00000000e+00 ,1.71305703e+03 ,3.41735502e+02] ,
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
        CRF = np.load(os.path.join(self.crf_dir, 'CRF_cam.npy'))
        CRF = CRF[2:27]
        return CRF
        
        