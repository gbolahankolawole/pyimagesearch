#OPENCV: Image Manipulation
import imutils
import cv2

image = 'resources/faces_8.png'

#load the image and print out the dimensions. DEPTH is the number of colors
img = cv2.imread(image)
(h, w, d) = img.shape
print ("width = {}, height = {}, depth = {}".format(w,h,d))

#display the image and any key on the keyboard when done
cv2.imshow("Image", img)
cv2.waitKey(0)

#access individual pixel. the image is a giant array of pixel values. Note OpenCV uses the old BGR notation
(B, G, R) = img[100,200]
print ("Blue = {}, Green = {}, Red = {}".format(B,G,R))

#crop out areas of interest. Note the args are y1:y2, x1:x2
roi = img[285:415, 1000:1120]
cv2.imshow("ROI: Young goofy Sakura", roi)
cv2.waitKey(0)

#Resizing image
resize_img = cv2.resize(img, (600,800))
cv2.imshow("Resized Image", resize_img)
cv2.waitKey(0)

#resize and preserve aspect ratio
new_width = 600
ratio = new_width / w
dim = (new_width, int(h * ratio))
resize_fix = cv2.resize(img, dim)
cv2.imshow("Resized Image Fixed", resize_fix)
cv2.waitKey(0)

#resize and preserve aspect ratio with imutils
resize_imutils = imutils.resize(img, width = 600)
cv2.imshow("Resized Image with imutils", resize_imutils)
cv2.waitKey(0)

#rotate image
center = (w // 2, h //2)
M = cv2.getRotationMatrix2D (center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w,h))
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)

#rotate image with imutils
rotated_imutils = imutils.rotate(img, 45)
cv2.imshow("Rotated Image with imutils", rotated_imutils)
cv2.waitKey(0)

#rotate image with imutils and dont crop image
rotated_imutils = imutils.rotate_bound(img, 45)
cv2.imshow("Rotated Image with imutils", rotated_imutils)
cv2.waitKey(0)

#image smoothening with Gaussian
blurred = cv2.GaussianBlur(img, (11,11), 0)
cv2.imshow("Smoothed Image", blurred)
cv2.waitKey(0)

#draw different shapes and write text to image. Use negative numbers (last argument in the functions called) to fill the shape 
output_img = img.copy()
cv2.rectangle(output_img, (610,100), (810,270), (200,50,50), 4)
cv2.circle(output_img, (685,395), 70, (200,50,50), 4)
cv2.line(output_img, (699,188), (691,393), (200,50,50), 4)
cv2.putText(output_img, "Connect two noses :)", (586,84), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (200,50,50), 1)

cv2.imshow("Shapaes and Text", output_img)
cv2.waitKey(0)
