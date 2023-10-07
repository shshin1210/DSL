import datetime
import numpy as np
from os.path import join

# scene parameters
SCENE_PATH = './experiments'
SCENE_NAME = 'test'
LOG_FN = SCENE_PATH + '/logs/' + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M') + '.txt'

# screen index
CAMERA_NUM = 0  # camera num
SCREEN_NUM = 1  # for projector

# debug
VERBOSE = True
SAVE_INTERM = False

# camera
PATTERN_PATH = './patterns/graycode_pattern/'
CAPTURE_WAIT_TIME = 0.3  #  0.2 unit: sec
SHUTTER_TIME = 45 # 60  # 600 # 100 # 500 in ms # 100 hyp 촬영
GAIN = 0  # this is log scale. /20 for conversion. 7.75db * 20, 0db - previous capture, for material capture
BINNING_RADIUS = 2
