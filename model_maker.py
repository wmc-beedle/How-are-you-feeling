import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from tqdm import tqdm
from skleran.model_selection import train_test_split
from image_pre import find_faces

# Limit number of predictions to increase accuracy
emotions = ['Peace','Confidence', 'Happiness','Doubt/Confusion', 'Fatigue', 'Embarrassment', 'Anger', 'Sadness', 'Fear', 'Pain']

f = pd.read_csv('emotic_pre/original_train.csv')

# Reduce the data
X_train, Y_train = find_faces(pd.read_csv('emotic_pre/original_train.csv')
X_test, Y_test = find_faces(pd.read_csv('emotic_pre/original_test.csv'))


model = keras.Sequential()
model.add(keras.layers.Conv2D(16,(3, 3), activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2, 2))


model.add(keras.layers.Conv2D(32,(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))

model.add(Flatten())


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tqdm(model.fit(X_train, Y_train, epoch=10, batch_size=16), desc='Loading Model...')
tqdm(evaluation = model.evaluate(X_test, Y_test), desc='Evaluating Model')
print('{}: {}'.format(model.metric_names[1], evaluation[1]*100))

model.save('emotion_detector.model')