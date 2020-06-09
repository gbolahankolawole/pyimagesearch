#OPENCV: Image Manipulation
import imutils
import cv2

image = 'resources/object_1.jpg'

#load the image and print out the dimensions
img = cv2.imread(image)
img = imutils.resize(img, width = 500)
cv2.imshow("Original Image", img)
cv2.waitKey(0)

#convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey Image", gray_img)
cv2.waitKey(0)

#detect edges
edges = cv2.Canny(gray_img, 120, 150)
cv2.imshow("Edges in the Image", edges)
cv2.waitKey(0)

#threshold the images
thresh = cv2.threshold(gray_img, 225,255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

#detect and draw contours and add descriptive text
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
output_img = img.copy()

for contour in contours:
	cv2.drawContours(output_img, [contour], -1, (50, 50, 200), 4)
	cv2.imshow("Contours detected in the Image", output_img)
cv2.waitKey(0)

text = "There are {} objects found in the image".format(len(contours))
cv2.putText(output_img, text, (10,25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (50,50,200), 2)
cv2.imshow("Objects found", output_img)
cv2.waitKey(0)

#erode, dilate and mask with bitwsie operations
mask = thresh.copy()

erode = cv2.erode(mask, None, iterations = 5)
cv2.imshow("Eroded Image", erode)
cv2.waitKey(0)

dilate = cv2.dilate(mask, None, iterations = 5)
cv2.imshow("Dilated Image", dilate)
cv2.waitKey(0)

masked_img = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow("Masked Image", masked_img)
cv2.waitKey(0)