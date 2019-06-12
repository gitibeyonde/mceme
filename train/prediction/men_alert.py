#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import sys
import time


if __name__ == "__main__":
    img = cv2.imread('/Users/abhi/work/mceme/ai/train/prediction/men.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    sys.exit(0)

