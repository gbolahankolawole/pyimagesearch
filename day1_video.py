#Detection of Faces with OpenCV

from imutils.video import VideoStream
import imutils
import time
import numpy as np
import cv2

def detectFaces(image):
    #Load the input image and construct an input blob for the image
    #by resizing to a fixed 300x300 pixels and then normalizing it
    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

    #pass the blob throught the network and obtain detections and predictions

    #print('[INFO]: computing object detection...')
    net.setInput(blob)
    detections = net.forward()

    #loop over detections

    for i in range (0, detections.shape[2]):
        #extract the confidence (i.e. the probability) associated with the prediction
        confidence = detections [0,0,i,2]
    
        #filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > confidence_threshhold:
            #compute the (x,y)-coordinates of the boinding box for the object
            box = detections [0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            #draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence*100)
            y = startY-10 if startY-10 > 10 else startY+10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

    return image

prototxt = 'resources/deploy.prototxt.txt'
model = 'resources/res10_300x300_ssd_iter_140000.caffemodel'
confidence_threshhold = 0.5

print ('[INFO]: loading model...')
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#initialize the camera and allow the sensor to warmup
print ('[INFO]: starting camera...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop over frames from video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1200)
    image = detectFaces(frame)


    #show output image
    cv2.imshow("Output", image)
    key = cv2.waitKey(1) & 0xFF

    #quit the loop if key `q' is pressed
    if key == ord("q"):
        break

#clean up resources
cv2.destroyAllWindows()
vs.stop()
