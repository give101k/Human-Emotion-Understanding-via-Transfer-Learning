import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
from glob import glob
import os

dirs = glob("../test_data/*/*/")
model = tf.keras.models.load_model("new resnet 50.model")
classes = ["Anger", "Happiness", "Neutral", "Sadness", "Surprise"]

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

IMG_SIZE = 224

for dir in dirs:
    image_paths = os.listdir(dir)
    j = 0
    for img_path in image_paths:
        if j > 20:
            break
        j += 1
        img = image.load_img(dir + img_path,
                             target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        index = preds.argmax(axis=-1)
        print('Predicted:', classes[index[0]])
        i = cv2.imread(dir + img_path)
        i_mess = cv2.putText(i, classes[index[0]], org, font,
                             fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('image', i_mess)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()
