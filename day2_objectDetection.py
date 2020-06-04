#OPENCV: Image Manipulation
import imutils
import cv2

image = 'resources/faces_8.png'

#load the image and print out the dimensions. DEPTH is the number of colors
img = cv2.imread(image)