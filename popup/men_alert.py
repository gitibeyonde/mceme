import cv2
import sys
import time


if __name__ == "__main__":
    img = cv2.imread('C:\\Users\\siddharth\\Documents\\ai\\popup\\men.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    sys.exit(0)
