import numpy as np
import cv2

vidcap = cv2.VideoCapture('../../data/vid/munchkin/munch-sup.mov')


def getFrame(sec):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop = gray[y-50:y + h+50, x:x + w]
            cv2.imwrite("../data/munch/surprise/munch-" +
                        str(count)+".jpg", crop)
    return hasFrames


sec = 0
frameRate = 0.03
count = 1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
