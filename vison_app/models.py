
import json
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
# Create your models here.

def load_model_from_path():
    graph = tf.get_default_graph()
    model = load_model('Vi_project/23Age_Gender_model.h5')
    return graph,model

def read_photo(path):
    imm = cv2.imread(path,cv2.IMREAD_COLOR)
    if imm.any():
        gray = cv2.cvtColor(imm, cv2.COLOR_RGB2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(60, 60)
                            )
        if len(faces)==1:
            for (x, y, w, h) in faces:
                cv2.rectangle(imm, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = imm[y:y + h, x:x + w]
                return roi_color
        elif len(faces)==0:
            return("No_face")
        else:
            return("Multiple_face")
    return None


def predition_age_gender(X):
    graph,model = load_model_from_path()
    X_data =[]
    face = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face, (32,32) )
    X_data.append(face)
    X_data=np.array(X_data)
    try:
        with graph.as_default():
            predictions = model.predict(X_data)
            gender = predictions[0]
            gender = "Female" if gender>0.5 else "Male"
            age = int(predictions[1])
            return gender,age
    except Exception as err:
        raise(err)
    return None
