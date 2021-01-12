import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from tqdm import tqdm

# different = ['Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
#              'Confidence', 'Happiness', 'Pleasure', 'Excitement','Surprise',
#              'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue', 'Embarrassment',
#              'Yearning', 'Disapproval', 'Aversion', 'Annoyance', 'Anger', 
#              'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain', 'Suffering']

emotions = ['Peace','Confidence', 'Happiness','Doubt/Confusion', 'Fatigue', 'Embarrassment', 'Anger', 'Sadness', 'Fear', 'Pain']

# f = pd.read_csv('./emotic_pre/original_train.csv', usecols=['Index','Folder','Filename','Categorical_Labels'])



face_det1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_det2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_det3 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_det4 = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

def find_faces(file):
    
    
    f = pd.read_csv(file,usecols=['Index','Folder','Filename','Categorical_Labels'])
    
    images = len(f.index)
    
    index = 0

    X = []
    Y = []
    
    for image in tqdm(range(images), desc='Loading...'):

        # X - image Processor

        frame = cv2.imread('./emotic/{}'.format(f['Folder'][image]) +'/{}'.format(f['Filename'][image]))
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face1 = face_det1.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = face_det2.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = face_det3.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = face_det4.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(face1) >= 1:
            features = face1
        elif len(face2) >= 1:
            features = face2
        elif len(face3) >= 1:
            features = face3
        elif len(face4) >= 1:
            features = face4
        else:
            features = ''
        for (x, y, w, h) in features:
            grayscale = grayscale[y:y+h, x:x+w]
            try:
                resized_image = cv2.resize(grayscale, (28, 28))
                for emotion in f['Categorical_Labels'][image].strip('][').split(', '):
                    emotion = emotion.replace("'","")
                    if emotion in emotions:
                        X.append(resized_image)
                        Y.append(emotion)
                    else:
                        continue
            except:
                pass
        index += 1 
    return X, Y
    print('X and Y are now loaded!')
find_faces('./emotic_pre/original_train.csv')