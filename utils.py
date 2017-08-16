import cv2
import time
import numpy as np
from math import pi

VIVE_FOV = 110
class central_peripheral_seperator():
    def __init__(self, FOV, frame_shape):
        self.central_dis = frame_shape[1] * (np.tan(pi * FOV / 360) / np.tan(pi * VIVE_FOV / (360))) / 2    
        self.center = (frame_shape[0] / 2, frame_shape[1] / 2)
        print self.central_dis
        self.choose_arr = np.zeros(frame_shape, dtype='int')

        for i in range(frame_shape[0]):
            for j in range(frame_shape[1]):
                if np.sqrt((i - self.center[0]) ** 2 + (j - self.center[1]) ** 2) <= self.central_dis:

                    self.choose_arr[i, j] = 1
            
    def seperate(self, frame):
        peripheral = np.zeros_like(frame)
        peripheral = np.choose(self.choose_arr, [frame, np.zeros_like(frame)])
        central = np.zeros_like(frame)
        central = np.choose(self.choose_arr, [np.zeros_like(frame), frame])

        return central, peripheral 

if __name__ == '__main__':
    cap = cv2.VideoCapture('test.mp4')
    
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 60, 3, 5, 1.2, 0)
    sep = central_peripheral_seperator(60, frame1[...,0].shape)
    central_flow, peripheral_flow = sep.seperate(flow)
    #print central_flow
    #print peripheral_flow
    
    #print frame1.shape
    mag_central, ang_central = cv2.cartToPolar(central_flow[...,0], central_flow[...,1])
    mag_peripheral, ang_peripheral = cv2.cartToPolar(peripheral_flow[...,0], peripheral_flow[...,1])

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    hsv[...,0] = ang_central*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = np.clip(mag_central * 5, 0, 255)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('central.png', bgr)

    hsv[...,0] = ang_peripheral*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = np.clip(mag_peripheral * 5, 0, 255)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('peripheral.png', bgr)

