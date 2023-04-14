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

# epipolar
CAPTURE_EPIPOLAR = False
CAPTURE_NONEPIPOLAR = False
CAPTURE_NEIGHBOUR_NONEPIPOLAR = False
CAPTURE_PLANE = False
CAPTURE_GRAYCODE = True 

PRJ_FN = './calibration/prj_inpainted.png'  # calibration file
CAM_FN = './calibration/cam_inpainted.png'  # calibration file
# ----- normal mode -----
NUM_SCAN_ROWS = 100
# ----- polar epi graycode mode -----
# NUM_SCAN_ROWS = 40 # 2 => 30min at 946*640, 
NUM_GRAYCODE = 42

# projector
PRJ_PATTERN_SCALAR = 1
# PRJ_PATTERN_SCALAR = 0.004

# debug
VERBOSE = True
SAVE_INTERM = False

# camera
PATTERN_PATH = './patterns/graycode_pattern/'
CAPTURE_WAIT_TIME = 0.3  #  0.2 unit: sec
SHUTTER_TIME = 60 # 60  # 600 # 100 # 500 in ms
GAIN = 0  # this is log scale. /20 for conversion. 7.75db * 20, 0db - previous capture, for material capture

ROI_HEIGHT = 28#100
SENSOR_HEIGHT = 1536
SENSOR_WIDTH = 2048
CAM_DO_CROP = False
CROP_OFFSET_X = 96 # 360  # 360
CROP_OFFSET_Y = 672 # 750  # 750
CROP_HEIGHT = 1244 #1264 # 700  # original: 900
CROP_WIDTH = 1456 #1664 # 1200  # original: 1200
if not CAM_DO_CROP:
    CROP_OFFSET_X = 0
    CROP_OFFSET_Y = 0
    CROP_HEIGHT = SENSOR_HEIGHT
    CROP_WIDTH = SENSOR_WIDTH

START_IND = 1 # This should be in the range of [1, NUM_RETARDER]

path = './ellipsometry_angles'
