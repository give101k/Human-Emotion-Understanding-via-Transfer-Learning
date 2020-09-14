import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input

cap = cv2.VideoCapture(0)
IMG_SIZE = 224
i = 0
while i < 1000:
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i += 1
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cropgray = gray[0:1080, 280:1000]
    #resized = cv2.resize(cropgray, (IMG_SIZE, IMG_SIZE))
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        crop = frame[y-50:y + h+50, x:x + w]
        #crop = preprocess_input(crop)
        if i > 700:
            cv2.imwrite("../test_data/Val/surprise/me " +
                        str(i) + ".jpg", crop)
        else:
            cv2.imwrite("../test_data/Train/surprise/me " +
                        str(i) + ".jpg", crop)
    print(i)
    cv2.imshow('frame', crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
