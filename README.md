# How are you feeling?
---
In this day and age, AI

# Goal

Make a model that when fed an image can find a face and estimate as to what emotion the individual is experiencing.


# Dependencies

* Pandas
* Numpy
* Sklearn
* OpenCV 
* Tensorflow
* tqdm
* ast

# [Emotic Dataset](http://sunai.uoc.edu/emotic/)

![Emotic_logo](Images/logo_emotic_v3.png)


*The EMOTIC dataset, named after EMOTions In Context, is a database of images with people in real environments, annotated with their apparent emotions. The images are annotated with an extended list of 26 emotion categories combined with the three common continuous dimensions Valence, Arousal and Dominance.*

*R. Kosti, J.M. Álvarez, A. Recasens and A. Lapedriza, "Context based emotion recognition using emotic dataset", IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2019.*

This dataset already comes in three parts: training, testing, and valuation sets. Each file has 9 columns with the training set having 23,265 rows, testing having 7,202 rows, and validation having 3,314 rows. The file used to preprocess the annotations from mat to csv file will be added back on once recovered.



# CNN
![Example_CNN](Images/Face-Recognition-CNN-Architecture.png)

*The image above is purely for explanation purposes of how the model works and not representative of the model itself.*


# Results
![sad](Images/face_emotion.png)