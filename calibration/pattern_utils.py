import cv2
import os
import numpy as np

def prepare_gray_pattern(intensity_uint8, size=[768, 1024]):
    blank_image = np.zeros(size, np.uint8)
    blank_image[:] = intensity_uint8
    pattern = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)

    return pattern


def prepare_pattern_list(mode = 3):
    if mode == 0:
        pattern_fns = [
            "patterns/gray/0.png",
            "patterns/gray/1.png",
            "patterns/gray/2.png",
            "patterns/gray/3.png",
            "patterns/gray/4.png",
            "patterns/gray/5.png",
            "patterns/gray/6.png",
            "patterns/gray/7.png",
            "patterns/gray/8.png",
            "patterns/gray/9.png",
            "patterns/gray/10.png",
            "patterns/gray/11.png",
            "patterns/gray/12.png",
            "patterns/gray/13.png",
            "patterns/gray/14.png",
            "patterns/gray/15.png",
            "patterns/gray/16.png",
            "patterns/gray/17.png",
            "patterns/gray/18.png",
            "patterns/gray/19.png",
            "patterns/gray/20.png",
            "patterns/gray/21.png",
            "patterns/gray/22.png",
            "patterns/gray/23.png",
            "patterns/gray/24.png",
            "patterns/gray/25.png",
            "patterns/gray/26.png",
            "patterns/gray/27.png",
            "patterns/gray/28.png",
            "patterns/gray/29.png",
            "patterns/gray/30.png",
            "patterns/gray/31.png",
            "patterns/gray/32.png",
            "patterns/gray/33.png",
            "patterns/gray/34.png",
            "patterns/gray/35.png",
            "patterns/gray/36.png",
            "patterns/gray/37.png",
            "patterns/gray/38.png",
            "patterns/gray/39.png",
            "patterns/gray/40.png",
            "patterns/gray/41.png",
            "patterns/gray/42.png",
            "patterns/gray/43.png"
        ]
    elif mode == 1:
        pattern_fns = [
            "patterns/gray_684_608/0.png",
            "patterns/gray_684_608/1.png",
            "patterns/gray_684_608/2.png",
            "patterns/gray_684_608/3.png",
            "patterns/gray_684_608/4.png",
            "patterns/gray_684_608/5.png",
            "patterns/gray_684_608/6.png",
            "patterns/gray_684_608/7.png",
            "patterns/gray_684_608/8.png",
            "patterns/gray_684_608/9.png",
            "patterns/gray_684_608/10.png",
            "patterns/gray_684_608/11.png",
            "patterns/gray_684_608/12.png",
            "patterns/gray_684_608/13.png",
            "patterns/gray_684_608/14.png",
            "patterns/gray_684_608/15.png",
            "patterns/gray_684_608/16.png",
            "patterns/gray_684_608/17.png",
            "patterns/gray_684_608/18.png",
            "patterns/gray_684_608/19.png",
            "patterns/gray_684_608/20.png",
            "patterns/gray_684_608/21.png",
            "patterns/gray_684_608/22.png",
            "patterns/gray_684_608/23.png",
            "patterns/gray_684_608/24.png",
            "patterns/gray_684_608/25.png",
            "patterns/gray_684_608/26.png",
            "patterns/gray_684_608/27.png",
            "patterns/gray_684_608/28.png",
            "patterns/gray_684_608/29.png",
            "patterns/gray_684_608/30.png",
            "patterns/gray_684_608/31.png",
            "patterns/gray_684_608/32.png",
            "patterns/gray_684_608/33.png",
            "patterns/gray_684_608/34.png",
            "patterns/gray_684_608/35.png",
            "patterns/gray_684_608/36.png",
            "patterns/gray_684_608/37.png",
            "patterns/gray_684_608/38.png",
            "patterns/gray_684_608/39.png",
            "patterns/gray_684_608/40.png",
            "patterns/gray_684_608/41.png"
        ]
    elif mode == 2:
        pattern_fns = [
            "patterns/checkerboard.png"
        ]
    elif mode == 3:
        pattern_fns = [
            "patterns/graycode_pattern/pattern_00.png",
            "patterns/graycode_pattern/pattern_01.png",
            "patterns/graycode_pattern/pattern_02.png",
            "patterns/graycode_pattern/pattern_03.png",
            "patterns/graycode_pattern/pattern_04.png",
            "patterns/graycode_pattern/pattern_05.png",
            "patterns/graycode_pattern/pattern_06.png",
            "patterns/graycode_pattern/pattern_07.png",
            "patterns/graycode_pattern/pattern_08.png",
            "patterns/graycode_pattern/pattern_09.png",
            "patterns/graycode_pattern/pattern_10.png",
            "patterns/graycode_pattern/pattern_11.png",
            "patterns/graycode_pattern/pattern_12.png",
            "patterns/graycode_pattern/pattern_13.png",
            "patterns/graycode_pattern/pattern_14.png",
            "patterns/graycode_pattern/pattern_15.png",
            "patterns/graycode_pattern/pattern_16.png",
            "patterns/graycode_pattern/pattern_17.png",
            "patterns/graycode_pattern/pattern_18.png",
            "patterns/graycode_pattern/pattern_19.png",
            "patterns/graycode_pattern/pattern_20.png",
            "patterns/graycode_pattern/pattern_21.png",
            "patterns/graycode_pattern/pattern_22.png",
            "patterns/graycode_pattern/pattern_23.png",
            "patterns/graycode_pattern/pattern_24.png",
            "patterns/graycode_pattern/pattern_25.png",
            "patterns/graycode_pattern/pattern_26.png",
            "patterns/graycode_pattern/pattern_27.png",
            "patterns/graycode_pattern/pattern_28.png",
            "patterns/graycode_pattern/pattern_29.png",
            "patterns/graycode_pattern/pattern_30.png",
            "patterns/graycode_pattern/pattern_31.png",
            "patterns/graycode_pattern/pattern_32.png",
            "patterns/graycode_pattern/pattern_33.png",
            "patterns/graycode_pattern/pattern_34.png",
            "patterns/graycode_pattern/pattern_35.png",
            "patterns/graycode_pattern/pattern_36.png",
            "patterns/graycode_pattern/pattern_37.png",
            "patterns/graycode_pattern/pattern_38.png",
            "patterns/graycode_pattern/pattern_39.png",
            "patterns/graycode_pattern/pattern_40.png",
            "patterns/graycode_pattern/pattern_41.png"
        ]
    for fn in pattern_fns:
        if not os.path.isfile(fn):
            print("[%s] not exist" % (fn))

    patterns = [cv2.imread(fn) for fn in pattern_fns]

    return patterns
