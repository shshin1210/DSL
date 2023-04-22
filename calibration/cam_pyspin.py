import numpy as np
import sys
import PySpin
import constants
import cv2

# import hdr #for testing
# register
# from logger import *
import matplotlib.pyplot as plt
import concurrent.futures as futures

# exposure = shutter

def configure_cam(cam, pixel_format, exposure_time_to_set_us,binning_radius, roi=None):
    # pixel_format: 'bayer gb16'
    # roi [ x, w, y, h ] -> [x, y, w, h]
    if roi is not None:
        w = roi[1]
        y = roi[2]
        roi[1] = y
        roi[2] = w

    # Initialize camera
    # cam.Init()

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()

    # Image format
    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
        # Retrieve the desired entry node from the enumeration node
        node_pixel_format_cur = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(pixel_format))  
        if PySpin.IsAvailable(node_pixel_format_cur) and PySpin.IsReadable(node_pixel_format_cur):
            # Retrieve the integer value from the entry node
            pixel_format_cur = node_pixel_format_cur.GetValue()
            # Set integer as new value for enumeration node
            node_pixel_format.SetIntValue(pixel_format_cur)
            print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())
        else:
            print('Pixel format %s not available...'%pixel_format)
    else:
        print('Pixel format not available...')

    # White balancing    
    # Auto off
    cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
    
    node_white_balance_format = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
    balance_ratio_node = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
    
    # Blue
    if PySpin.IsAvailable(node_white_balance_format) and PySpin.IsWritable(node_white_balance_format):
        # Retrieve the desired entry node from the enumeration node
        node_white_balance_format_cur = PySpin.CEnumEntryPtr(node_white_balance_format.GetEntryByName("Blue")) 
        if PySpin.IsAvailable(node_white_balance_format_cur) and PySpin.IsReadable(node_white_balance_format_cur):
            # Retrieve the integer value from the entry node
            # Set integer as new value for enumeration node
            balance_ratio_node.SetValue(1.0)
            print('WB blue set to %s...' % balance_ratio_node.GetValue())
        else:
            print('WB %s not available...'%("blue"))
    else:
        print('WB not available...')
        
    
    # Red
    # cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
    
    node_white_balance_format = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
    if PySpin.IsAvailable(node_white_balance_format) and PySpin.IsWritable(node_white_balance_format):
        # Retrieve the desired entry node from the enumeration node
        node_white_balance_format_cur = PySpin.CEnumEntryPtr(node_white_balance_format.GetEntryByName("Red"))  
        if PySpin.IsAvailable(node_white_balance_format_cur) and PySpin.IsReadable(node_white_balance_format_cur):
            # Retrieve the integer value from the entry node
            # Set integer as new value for enumeration node
            balance_ratio_node.SetValue(1.0)
            print('WB red set to %s...' % balance_ratio_node.GetValue())
        else:
            print('WB %s not available...'%("red"))
    else:
        print('WB not available...')

    
    # Set ROI first if applicable (framerate limits depend on it)
    # try:
    # Note set width/height before x/y offset because upon
    # initialization max offset is 0 because full frame size is assumed
    if roi is not None:
        for i, (s, o) in enumerate(zip(["Width", "Height"], ["OffsetX", "OffsetY"])):

            roi_node = PySpin.CIntegerPtr(nodemap.GetNode(s))
            # If no ROI is specified, use full frame:
            if roi[2 + i] == -1:
                value_to_set = roi_node.GetMax()
            else:
                value_to_set = roi[2 + i]
            inc = roi_node.GetInc()
            if np.mod(value_to_set, inc) != 0:
                value_to_set = (value_to_set // inc) * inc
            roi_node.SetValue(value_to_set)

            # offset
            offset_node = PySpin.CIntegerPtr(nodemap.GetNode(o))
            if roi[0 + i] == -1:
                off_to_set = offset_node.GetMin()
            else:
                off_to_set = roi[0 + i]
            offset_node.SetValue(off_to_set)

        # except Exception as ex:
        #     print("E:Could not set ROI. Exception: {0}.".format(ex))

    # Turn off Auto Gain
    node_gainauto_mode = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    node_gainauto_mode_off = node_gainauto_mode.GetEntryByName("Off")
    node_gainauto_mode.SetIntValue(node_gainauto_mode_off.GetValue())


    # Set gain to 0 dB
    node_iGain_float = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    node_iGain_float.SetValue(0)

    # Turn on Gamma
    # node_gammaenable_mode = PySpin.CBooleanPtr(nodemap.GetNode("GammaEnable"))
    # # node_gammaenable_mode.SetValue(True)
    # node_gammaenable_mode.SetValue(Value=True, Verify=True)

    # Set Gamma as 1
    node_Gamma_float = PySpin.CFloatPtr(nodemap.GetNode("Gamma"))
    node_Gamma_float.SetValue(1) # linear한 output

    # Configure exposure
    if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
        print('Unable to disable automatic exposure. Aborting...')
        return False
    
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    # Set exposure time manually; exposure time recorded in microseconds
    if cam.ExposureTime.GetAccessMode() != PySpin.RW:
        print('Unable to set exposure time. Aborting...')
        return False

    # set binning
    node_binning_vertical = PySpin.CIntegerPtr(nodemap.GetNode('BinningVertical'))
    node_binning_vertical.SetValue(binning_radius)


    # Ensure desired exposure time does not exceed the maximum
    exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set_us)
    # exposure_time_to_set = max(cam.ExposureTime.GetMax(), exposure_time_to_set_us)
    cam.ExposureTime.SetValue(exposure_time_to_set)
    print('Shutter time set to %s ms...\n' % (exposure_time_to_set/1e3))

    # Ensure trigger mode off
    # The trigger must be disabled in order to configure whether the source
    # is software or hardware.
    nodemap = cam.GetNodeMap()
    node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
    if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
        print('Unable to disable trigger mode (node retrieval). Aborting...')
        return False

    node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
    if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
        print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
        return False

    node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

    print('Trigger mode disabled...')
    

    # Set TriggerSelector to FrameStart
    # For this example, the trigger selector should be set to frame start.
    # This is the default for most cameras.
    node_trigger_selector= PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
    if not PySpin.IsAvailable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
        print('Unable to get trigger selector (node retrieval). Aborting...')
        return False

    node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
    if not PySpin.IsAvailable(node_trigger_selector_framestart) or not PySpin.IsReadable(
            node_trigger_selector_framestart):
        print('Unable to set trigger selector (enum entry retrieval). Aborting...')
        return False
    node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())
    
    print('Trigger selector set to frame start...')
    # Select trigger source
    # The trigger source must be set to hardware or software while trigger
    # mode is off.
    node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
    if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
        print('Unable to get trigger source (node retrieval). Aborting...')
        return False

    node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
    if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
            node_trigger_source_software):
        print('Unable to set trigger source (enum entry retrieval). Aborting...')
        return False
    node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
    print('Trigger source set to software...')

    # Turn trigger mode on
    # Once the appropriate trigger source has been set, turn trigger mode
    # on in order to retrieve images using the trigger.
    node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
    if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
        print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
        return False

    node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
    print('Trigger mode turned back on...')

    # Set acquisition mode to continuous
    # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
            node_acquisition_mode_continuous):
        print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        return False

    # Retrieve integer value from entry node
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    print('Acquisition mode set to continuous...')


def get_cam():

    # initialize camera
    # singleton reference to system obj
    system = PySpin.System.GetInstance()
    # list of cam from sys
    cam_list = system.GetCameras()
    # number of cam
    num_cameras = cam_list.GetSize()
    cam = cam_list[0]
    cam.Init()

    print(cam.DeviceModelName())
    return cam


    # setup camera
    

def trigger_on(cam):
    nodemap = cam.GetNodeMap()
    # Execute software trigger
    node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
    if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
        print('Unable to execute trigger. Aborting...')

    node_softwaretrigger_cmd.Execute()
    # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger


def timeout(timelimit):
    def decorator(func):
        def decorated(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timelimit)
                except futures.TimeoutError:
                    print('Timedout!')
                    raise TimeoutError from None
                else:
                    print(result)
                executor._threads.clear()
                futures.thread._threads_queues.clear()
                return result
        return decorated
    return decorator

def capture_im(cam, pixel_format, cv2_demosaic_format=cv2.COLOR_BayerGB2BGR):
    trigger_on(cam)
    im_raw = cam.GetNextImage(1000)
    # converting images, color processing algorithm
    im_raw_con = im_raw.Convert(PySpin.PixelFormat_BayerGB16, PySpin.HQ_LINEAR)
    im_raw_dat = im_raw_con.GetData()
    
    # Printing image information
    width = im_raw.GetWidth()
    height = im_raw.GetHeight()
    im_raw_dat = im_raw_dat.reshape((height, width, -1))

    im = cv2.demosaicing(im_raw_dat, cv2_demosaic_format)
    if im.dtype == np.uint16:
        im_result = im/65535
        
    im_raw.Release()
    
    return im_result

if __name__ == '__main__':

    # ===========================================================
    # cam = get_cam()
    system = PySpin.System.GetInstance()
    # list of cam from sys
    cam_list = system.GetCameras()
    # number of cam
    num_cameras = cam_list.GetSize()
    cam = cam_list[0]
    cam.Init()

    print(cam.DeviceModelName())
    
    pixel_format = 'BayerGB16'

    # setup camera
    configure_cam(cam, pixel_format, constants.SHUTTER_TIME*1e3, roi=None)

    # Begin acquiring images/ capturing images
    cam.BeginAcquisition()

    N_img = 2

    # capture
    for i in range(N_img):
        trigger_on(cam)
        # Now using cam[0]
        im_raw = cam.GetNextImage(1000)
        # converting images, color processing algorithm
        im_raw_con = im_raw.Convert(PySpin.PixelFormat_BayerGB16, PySpin.HQ_LINEAR)

        im_raw_dat = im_raw_con.GetData()
        
        # Printing image information
        width = im_raw.GetWidth()
        height = im_raw.GetHeight()
        im_raw_dat = im_raw_dat.reshape((height, width, -1))
        # im_raw_dat = im_raw_dat/65535
        im = cv2.demosaicing(im_raw_dat, cv2.COLOR_BayerGB2BGR)
        im_result = im/65535
        
        plt.figure()
        plt.imshow(im_result)
        plt.show()

        # cv2.imwrite(f'./im_{i}.png')
        
        im_raw.Release()
        
    cam.EndAcquisition()
    
    cam.DeInit()
    del cam
    
    cam_list.Clear()
    system.ReleaseInstance()
        # releasing?
        # im_raw_dat.Release()
    