#DOCUMENT SCANNER
from pyimagesearch import four_point_transform
from skimage.filters import threshold_local
import numpy as numpy
import cv2
import imutils

image = 'resources/document_4.jpeg'

#load image
img = cv2.imread(image)
ratio = img.shape[0] / 500.0
original_img = img.copy()
img = imutils.resize(img, height = 500)

#convert ot gray scale, blur ir and find edges
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (5,5), 0)
edges = cv2.Canny(gray_img, 75, 200)

print("[INFO]: STEP 1 - Edge Detection")
cv2.imshow("Image", img)
cv2.imshow("Edges in Image", edges)
cv2.waitKey(0)

#find contours in the edged image, keep only the largest ones, and intialize screen contour
contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

#loop over contours
for contour in contours:
	#approximate the contour
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

	#if our approximated contour has four pionts, then we can assume that we have found the screen
	if len(approx) == 4:
		screenContour = approx
		break

#show the contour outline on the image
print("[INFO]: Step 2 - Finding contours in image")
cv2.drawContours(img, [screenContour], -1, (50, 50, 200), 3)
cv2.imshow("Outline Contours in Image", img)
cv2.waitKey(0)

#apply document transform to the image
warped = four_point_transform(original_img, screenContour.reshape(4,2) * ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

print ("INFO: STEP 3 - Apply perspective transform")
cv2.imshow("original", imutils.resize(original_img, height = 500))
cv2.imshow("scanned", imutils.resize(warped, height = 500))
cv2.waitKey(0)