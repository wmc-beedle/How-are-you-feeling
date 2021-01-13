import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from library import emotion2int, int2emotion, gender2int, int2gender, get_photo, make_numeric, featurize
from sklearn.model_selection import train_test_split

emotions = {'Peace':0, 'Affection':1, 'Esteem':2, 'Anticipation':3, 'Engagement':4,
             'Confidence':5, 'Happiness':6, 'Pleasure':7, 'Excitement':8,'Surprise':9,
             'Sympathy':10, 'Doubt/Confusion':11, 'Disconnection':12, 'Fatigue':13, 'Embarrassment':14,
             'Yearning':15, 'Disapproval':16, 'Aversion':17, 'Annoyance':18, 'Anger':19, 
             'Sensitivity':20, 'Sadness':21, 'Disquietment':22, 'Fear':23, 'Pain':24, 'Suffering':25}
genders = {'Female':0,'Male':1}

train_file = 'Data/emotic_pre/train.csv'
test_file = 'Data/emotic_pre/test.csv'

X, Y = featurize(train_file,test_file)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Sequential()
model.add(Conv2D(16,(3, 3), activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32,(3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(keras.layers.Dense(4))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=4)
evaluation = model.evaluate((X_test, Y_test), desc='Evaluating Model...')
# print('{}: {}'.format(model.metric_names[1], evaluation[1]*100))
start_time = timeit.default_timer()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
elapsed = timeit.default_timer() - start_time
print("\nTest data, accuracy: {:5.2f}%".format(100*test_acc))
print('\nTook {:.2f}s to finish'.format(elapsed))
# model.save('emotion_detector.model')