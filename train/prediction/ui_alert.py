#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image

import logging
import signal
import os
import urllib
import numpy as np
import matplotlib.pyplot as plt
import circularQueue
import sched, time

log_level = logging.DEBUG
logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)

ALERT_FILE='/root/alert.txt'
SNAPSHOT_URL='https://dummyimage.com/600x400/000/fff'

imageList = circularQueue.circularQueue(5)

def gen_alert(sig, stack):
    st, et, alert = get_alert()
    logging.debug("ALERT RECEIVED ===  St = %s, Et = %s Alert = %s"%(st, et, alert))
    

signal.signal(signal.SIGUSR2, gen_alert)

def get_alert():
    if os.path.exists( ALERT_FILE ) and os.path.getsize( ALERT_FILE ) > 0:
        with open(ALERT_FILE) as f: alert = f.read().strip()
        return alert.split(":")
    else:
        return 1, 2, 'ONE'

def download_image(sc): 
    logging.debug("Downloading image .. ")
    imgpng = Image.open(urllib.urlopen(SNAPSHOT_URL))
    ImageNumpyFormat = np.asarray(imgpng)
    imageList.enqueue(ImageNumpyFormat)
    sc.enter(1, 1, download_image, (s,))
    
def draw_image():
    ImageNumpyFormat = imageList.dequeue()
    plt.imshow(ImageNumpyFormat)
    plt.draw()
    plt.pause(15) # pause how many seconds
    plt.close()
    
    
if __name__ == "__main__":
    s = sched.scheduler(time.time, time.sleep)
    logging.debug("Schedular created .. ")
    s.enter(1, 1, download_image, (s,))
    logging.debug("Schedular started .. ")
    s.run()