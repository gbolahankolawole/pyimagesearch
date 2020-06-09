#four point transform
import numpy as np
import cv2

def order_points(pts):
	#intialise a list of coordinates that will be ordered from top left clock wise to bottom left
	rect = np.zeros((4,2), dtype = "float32")

	#the top-left point will have the smallest sum whereas the bottom-right will has the largest sum
	s = pts.sum(axis=1)
	rect [0] = pts [np.argmin(s)]
	rect [2] = pts [np.argmax(s)]

	#the top-right point will have the smallest diff whereas the bottom-left will has the largest diff
	diff = np.diff(pts, axis =1)
	rect[1] = pts [np.argmin(diff)]
	rect[3] = pts [np.argmax(diff)]

	return rect

def four_point_transform(img, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(
					((br[0] - bl[0]) ** 2) 
					+ ((br[1] - bl[1]) ** 2)
					)
	widthB = np.sqrt(
					((tr[0] - tl[0]) ** 2) 
					+ ((tr[1] - tl[1]) ** 2)
					)
	maxWidth = max(int(widthA), int(widthB))


	heightA = np.sqrt(
					((tr[0] - br[0]) ** 2) 
					+ ((br[1] - br[1]) ** 2)
					)
	heightB = np.sqrt(
					((tl[0] - bl[0]) ** 2) 
					+ ((tl[1] - bl[1]) ** 2)
					)
	maxHeight = max(int(heightA), int(heightB))


	dst = np.array(
		[
			[0,0],
			[maxWidth -1, 0],
			[maxWidth -1, maxHeight -1],
			[0, maxHeight -1]
		],
		dtype = "float32"
		)

	M = cv2.getPerspectiveTransform(rect,dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

	return warped
