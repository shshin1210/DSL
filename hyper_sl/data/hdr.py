from hyper_sl.utils.ArgParser import Argument
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import cv2, os, sys
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate
import torch

"""
    Create HDR image and create PPG graph
"""

class HDR():
    def __init__(self, arg):

        self.cam_H, self.cam_W = arg.cam_H, arg.cam_W
        
            
        print('make hdr image')
        
if __name__ == "__main__":
        
    argument = Argument()
    arg = argument.parse()
    
