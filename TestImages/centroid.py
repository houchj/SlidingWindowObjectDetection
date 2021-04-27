import numpy as np
import cv2
from matplotlib import pyplot as plt
import queue
from PIL import ImageGrab, Image, ImageOps

# Picture path
imgname = "centroid.png"


# grey scale image with black background
def get_centroid(image):
    m = cv2.moments(image)
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return x, y


def is_centroid_in_window(x, y, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x1 < x and x > x2) and (y1 < y and y > y2)


# Read picture
image = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
image = 255 - image
# Image height and width
h, w = image.shape[:2]
print(w, h)

x , y = get_centroid(image)

print("centroid is {}, {}".format(x, y))

cv2.circle(image, (int(x), int(y)), 5, 255, 5)
cv2.imshow('out', image)

#
cv2.waitKey(0)
cv2.destroyAllWindows()
