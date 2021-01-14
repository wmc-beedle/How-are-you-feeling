import pandas as pd 
import numpy as np
import os 
import cv2
import ast 
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# Categorical data to convert

emotions = {'Peace':0, 'Happiness':1, 'Surprise':2,'Doubt/Confusion':3, 
            'Fatigue':4, 'Embarrassment':4, 'Anger':5, 'Sadness':6, 'Fear':7, 'Pain':8}
genders = {'Female':0,'Male':1}
ages = {'Adult':0, 'Teenager':1, 'Kid':2}


# Needed to convert Categorical data into Numerical Data

# Emotion to  interger conversions
def emotion2int(emotion):
    lst = []
    for emot in [emotion].strip('][').split(', '):
                    emot = emotion.replace("'","")
    val = emotions[emot]
    return val

# Gender to integer conversions

def gender2int(gender):
    val = genders[gender]
    return val

def int2emotion(int):
    for key, value in emotions.items():
        if int == value:
            return key

# Age to interger conversio
def age2int(age):
    val = ages[age]
    return val

def int2gender(int):
    for key, value in genders.items():
        if int == value:
            return key


# To Retrieve the photo
def get_photo(file, idx):
    folder = file.iloc[idx, 1]
    filename = file.iloc[idx, 2]
    cwd = os.getcwd() + '/Data'

    img = image.load_img(f'{cwd}/emotic/{folder}/{filename}', target_size=(48,48,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255.0   
    return img

# Collect all the annotations
def make_numeric(file, idx):
    BB = file.iloc[idx, 3]
    Labels = emotion2int(file.iloc[idx, 5])
    Gender = gender2int(file.iloc[idx, 7])
    Age = file.iloc[idx, 8]
    num_list = np.array([BB,Labels,Gender,Age])
    return num_list


# Converter
def Converter(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    X = []
    Y = []

    for idx in tqdm(range(len(train)), desc='Loading training data...'):
        try:
            img_train = get_photo(train,idx)
            X.append(img_train)
            Y.append(make_numeric(train,idx))
        except:
            pass
    for idx in tqdm(range(len(test)), desc='Loading testing data...'):
        try:
            img_test = get_photo(test,idx)
            X.append(img_test)
            Y.append(make_numeric(test,idx))
        except:
            pass
    print('Loading data done!')
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
