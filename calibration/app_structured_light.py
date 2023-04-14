import cv2
import numpy as np
import pattern_utils
import time
import socket
import sys
import os
import pickle
import network
import cam_pyspin
import constants
import datetime
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import PySpin

# thread for capture image
class CaptureThread(QThread):
    '''
    Thread that projects a pattern, captures and save an image
    '''
    number_changed = pyqtSignal(int)
    image_changed = pyqtSignal(object)

    def __init__(self, scene_name):
        QThread.__init__(self)
        self.runs = True
        self.scene_name = scene_name
        self.output_dir = '%s/%s/' % (constants.SCENE_PATH, self.scene_name)
        self.is_finished = False

    def stop(self):
        self.runs = False

    def run(self):
        for i in range(len(patterns)):
            self.number_changed.emit(i)

            if not self.runs:
                return
            self.capture(i)
        self.is_finished = True

    def capture(self, i):
        pattern = patterns[i]

        self.image_changed.emit(pattern)
        global camera, pixel_format
        
        time.sleep(constants.CAPTURE_WAIT_TIME)
        im = cam_pyspin.capture_im(camera, pixel_format)
        time.sleep(constants.CAPTURE_WAIT_TIME)
        #cv_image = (im*65535).astype(np.uint16)
        cv_image = (im*256).astype(np.uint8)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        cv2.imwrite("%s/capture_%04d.png" % (self.output_dir, i + 1), cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
        # captured_images.append(cv_image)


class ClientThread(QThread):
    image_changed = pyqtSignal(object)

    def __init__(self, client):
        QThread.__init__(self)
        self.client = client

    def run(self):
        while True:
            recv = self.client.recv(network.BUFFER_SIZE)
            try:
                packet = pickle.loads(recv)
                pattern = patterns[packet.content]
                self.image_changed.emit(pattern)
            except EOFError:
                continue


class ServerThread(QThread):

    def __init__(self, server):
        QThread.__init__(self)
        self.server = server
        self.conns = []

    def run(self):
        while True:
            conn, addr = self.server.accept()
            self.conns.append(conn)

    def broadcast(self, message):
        # self.conns = [conn for conn in self.conns if not conn.closed]
        for conn in self.conns:
            conn.send(pickle.dumps(message))


# thread for camera preview
class PreviewThread(QThread):

    camera_captured = pyqtSignal(object)

    def __init__(self):
        QThread.__init__(self)
        self.do_capture = False

    def capture(self):
        self.do_capture = True

    def run(self):
        while True:
            if self.do_capture:
                global camera, pixel_format
                im = cam_pyspin.capture_im(camera, pixel_format)
                cv_image = (im * 256).astype(np.uint8)
                self.camera_captured.emit(cv_image)
                self.do_capture = False

# Image window
class ImageWindow(QLabel):
    def __init__(self, screen_num):
        QLabel.__init__(self)

        self.setScaledContents(True)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setGeometry(QApplication.desktop().screenGeometry(screen_num))
        self.showFullScreen()

    def show_image(self, image):
        height, width, channels = image.shape
        bytes_per_line = width * channels
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap(q_img))
        self.repaint()

# main window
class PreviewWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # setup ui
        self.preview = QLabel()
        self.preview.setPalette(palette)
        self.preview.setScaledContents(True)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.progress_bar = QProgressBar()

        self.textarea = QLabel()
        self.textarea.setWordWrap(True)
        self.textarea.setText("")

        vbox = QVBoxLayout()
        vbox.addWidget(self.preview, 1, Qt.AlignCenter)
        vbox.addWidget(self.progress_bar)
        vbox.addWidget(self.textarea)

        self.setLayout(vbox)

        self.setGeometry(100, 100, 360, 400)
        self.show()

        self.thread = PreviewThread()
        self.thread.camera_captured.connect(self.show_preview)
        self.thread.start()

        # network
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', network.TCP_PORT))
        self.server.listen(1)

        self.server_thread = ServerThread(self.server)
        self.server_thread.start()

        # image viewers
        self.labels = [ImageWindow(constants.SCREEN_NUM)]

        # capture handler
        self.capture_handler = CaptureHandler(self, constants.SCENE_NAME, self.labels)

    def set_capture_number(self, number):
        self.progress_bar.setValue(100 * (number + 1) / len(patterns))
        self.textarea.setText("Capturing... [Calibration: %02d, Capture number: %d / %d]" % (self.capture_handler.scene_number, (number + 1), len(patterns)))

    def set_scene_number(self, scene_number):
        self.textarea.setText("Next Calibration: %02d" % scene_number)

    def show_preview(self, image):
        height, width, channels = image.shape
        bytes_per_line = width * channels
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap(q_img))
        self.preview.repaint()

    def keyPressEvent(self, e):
        num_keys = [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9]
        indices = [0, 1, 2, 3, 4, 7, 15, 21, 31, 41]
        for i, num_key in zip(indices, num_keys):
            if e.key() == num_key:
                self.show_image_idx(i, 0)

        if e.key() == Qt.Key_P:
            self.thread.capture()

        if e.key() == Qt.Key_Escape:
            for label in self.labels:
                label.close()
            self.close()

        if e.key() == Qt.Key_Space:
            self.capture_handler.capture_sequence()

        if e.key() == Qt.Key_Q:
            self.release()

    def release(self):
        global camera, system, cam_list
        camera.EndAcquisition()
    
        camera.DeInit()
        del camera
        
        cam_list.Clear()
        system.ReleaseInstance()
        
    def closeEvent(self, e):
        for label in self.labels:
            label.close()
        self.close()

    def show_image_idx(self, image_idx, window_num):
        num_labels = len(self.labels)
        num_conns = len(self.server_thread.conns)

        if window_num < num_labels:
            self.labels[window_num].show_image(patterns[image_idx])
        elif window_num < (num_labels + num_conns):
            self.server_thread.conn[window_num - num_labels].send(
                network.Packet(network.PacketType.show_calib_image, image_idx))


class CaptureHandler:
    def __init__(self, preview_window, scene_name, labels):
        self.scene_number = 0
        self.scene_name = scene_name + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
        self.capture_thread = None
        self.preview_window = preview_window
        self.labels = labels

    def next_capture(self):
        self.scene_number += 1
        self.preview_window.set_scene_number(self.scene_number)

    def capture_sequence(self):
        if self.capture_thread is not None and not self.capture_thread.is_finished:
            return

        c = CaptureThread("%s/calibration%d" % (self.scene_name, self.scene_number))
        c.number_changed.connect(lambda idx: self.preview_window.show_image_idx(idx, 0))
        c.number_changed.connect(self.preview_window.set_capture_number)
        c.finished.connect(self.next_capture)
        c.start()

        self.capture_thread = c

# cameara
# ===========================================================
# camera = cam_pyspin.get_cam()

# initialize camera
# singleton reference to system obj
system = PySpin.System.GetInstance()
# list of cam from sys
cam_list = system.GetCameras()
# number of cam
num_cameras = cam_list.GetSize()
camera = cam_list[0]
camera.Init()

print(camera.DeviceModelName())


pixel_format = 'BayerGB16'

# setup camera
cam_pyspin.configure_cam(camera, pixel_format, constants.SHUTTER_TIME*1e3, roi = None)

# Begin acquiring images/ capturing images
camera.BeginAcquisition()
    
# load patterns
patterns = pattern_utils.prepare_pattern_list(mode = 3) # mode 1,2,3 (patterns)

# list for captured images
palette = QPalette()
palette.setColor(QPalette.Background, Qt.white)


if __name__ == '__main__':

    # prepare window
    app = QApplication(sys.argv) # app
    
    # setup ui
    previewWindow = PreviewWindow()
    
    # del camera
    
    # cam_list.Clear()
    
    # system.ReleaseInstance()
    
    sys.exit(app.exec_())