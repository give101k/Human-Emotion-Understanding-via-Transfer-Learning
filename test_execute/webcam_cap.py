import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model("new resnet 50.model")

cap = cv2.VideoCapture(0)
classes = ["anger", "hap", "neut", "sad", "surp"]
IMG_SIZE = 224

while(True):
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropgray = gray[0:1080, 280:1000]
    resized = cv2.resize(cropgray, (IMG_SIZE, IMG_SIZE))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop = gray[y-50:y + h+50, x:x + w]
        crop = preprocess_input(crop)
        cv2.imwrite("capimg.jpg", crop)

    img = image.load_img('capimg.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = utils.preprocess_input(x, version=1)  # or version=2
    preds = model.predict(x)
    index = preds.argmax(axis=-1)
    print('Predicted:', classes[index[0]])

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w+10, y+h+10), (0, 255, 0), 2)
        cv2.putText(frame, classes[index[0]], (x, y-10), font,
                    fontScale, color, thickness, cv2.LINE_AA)
    disimg = cv2.putText(frame, classes[index[0]], org, font,
                         fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('frame', disimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
