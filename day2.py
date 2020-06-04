#OPENCV: Image Manipulation and Object Detection

import imutils
import cv2

image = 'resources/faces_2.jpg'

#load the image and print out the dimensions. DEPTH is the number of colors
img = cv2.imread(image)
(h, w, d) = img.shape
print ("width = {}, height = {}, depth = {}".format(w,h,d))

#display the image and any key on the keyboard when done
cv2.imshow("Image", img)
cv2.waitKey(0)