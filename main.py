import numpy as np
import cv2

# Get the trained caffe model
# For image face detection
net = cv2.dnn.readNetFromCaffe('lib/deploy.prototxt',
                                            'lib/res10_300x300_ssd_iter_140000.caffemodel')

# Reading the image
frame = cv2.imread('test data/1641494771535.jpg')
# Get the Width & Height of the image
(h, w) = frame.shape[:2]
# Preprocess the image
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (124.0, 177.0, 150.0))
# Add the image preprocessed as an input to the model
net.setInput(blob)
# Get faces from the input image
faces = net.forward()
# If there's faces in the image
if len(faces) > 0:
    # Blurring image
    blurred = cv2.blur(frame, (30, 30))
    # Loop on the face(s) in the image
    for i in range(0, faces.shape[2]):
        # Get face accuracy
        confidence = faces[0, 0, i, 2]
        # If face accuracy less than 0.5 go to the next face
        if confidence < 0.5:
            continue
        else:
            # Get boundary box of this face
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            # Draw a rectangle on this face
            frame = cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
            # Get the blurred face (blurred boundary box) and add it to the original frame
            frame[startY:endY, startX:endX] = blurred[startY:endY, startX:endX]
    # Show the image
    cv2.imshow('output', frame)
    cv2.waitKey(0)
else: print("THERE'RE NO FACES IN THIS IMAGE !!!")