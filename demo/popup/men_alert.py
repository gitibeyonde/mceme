import cv2
import sys
import time

FILE_SEPARATOR='/'
#FILE_SEPARATOR = '\\'

if __name__ == "__main__":
    img = cv2.imread('popup'+FILE_SEPARATOR+'men.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    sys.exit(0)
