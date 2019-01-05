#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import signal
import os
import sys
import time


PID_FILE='/Users/abhi/work/mceme/ai/train/prediction/IMGALERT.pid'


def displayMen(sig, stack):
    img = cv2.imread('men.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    
  
def displayVehicle(sig, stack):
	img = cv2.imread('vehicle.jpg')
	cv2.imshow('image', img)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
      

signal.signal(signal.SIGUSR1, displayMen)
signal.signal(signal.SIGUSR2, displayVehicle)


if _name_ == "__main__":
    while True:
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            sys.exit(0) 
