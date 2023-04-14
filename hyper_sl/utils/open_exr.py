import os
import numpy as np
import sys
print(os.getcwd())
sys.path.append("C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging")
# from hyper_sl.utils import constants as C

import OpenEXR, Imath

class openExr():
    def __init__(self,arg):
    
        self.output_dir = arg.output_dir
        
    def read_exr_as_np(self, i, render_type):
        """  Read exr file(rgb rendered obj scene) as numpy array
        
        exr file shape RGBA 640x640x4
        return : numpy array        
        """
        
        fn = sorted(os.listdir(self.output_dir))
        fn = [file for file in fn if render_type in file]
        
        fn = os.path.join(self.output_dir, fn[i])

        f = OpenEXR.InputFile(fn)
        channels = f.header()['channels']

        dw = f.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        ch_names = []

        image = np.zeros((size[1], size[0], len(channels)-1))

        for i, ch_name in enumerate(channels):
            ch_names.append(ch_name)
            ch_dtype = channels[ch_name].type
            ch_str = f.channel(ch_name, ch_dtype)

            if ch_dtype == Imath.PixelType(Imath.PixelType.FLOAT):
                np_dtype = np.float32
            elif ch_dtype == Imath.PixelType(Imath.PixelType.HALF):
                np_dtype = np.half

            image_ch = np.fromstring(ch_str, dtype=np_dtype)
            image_ch.shape = (size[1], size[0])

            if ch_name == "A" :
                continue
            else:
                image[:,:,3-i] = image_ch
        
        return image
