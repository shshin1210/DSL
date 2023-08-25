import torch
import numpy as np
import os
from scipy.interpolate import interp1d
from hyper_sl.utils import calibrated_params

class Camera():
    def __init__(self, arg):
        self.arg = arg
        self.crf_dir = arg.camera_response
        
        # params
        self.cam_int, _ = calibrated_params.bring_params(arg.calibration_param_path,"cam")

    def intrinsic_cam(self):        
        intrinsic_cam = torch.tensor(self.cam_int).type(torch.float32).to(self.arg.device)
        
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
        CRF = np.load(os.path.join(self.crf_dir, 'CRF.npy')).T
        # CRF = np.load(os.path.join(self.crf_dir, 'CRF_cam.npy'))
        # map_scale = interp1d([CRF.min(), CRF.max()], [0.,1.])
        # CRF = torch.tensor(map_scale(CRF).astype(np.float32))        
        # CRF = CRF[2:27]
        return CRF
        
        