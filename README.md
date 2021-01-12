# How are you feeling?
---
In this day and age, cameras are everywhere

# Goal

Make a model that when fed an image can find a face and estimate as to what emotion the individual is experiencing.


# Needed to Run

* Pandas
* Numpy
* OpenCV - cv2
* Tensorflow
* tqdm

# Data 
[Emotic Dataset](http://sunai.uoc.edu/emotic/)

*R. Kosti, J.M. Álvarez, A. Recasens and A. Lapedriza, "Context based emotion recognition using emotic dataset", IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2019.*

This dataset is broken up into two different kind of measurments: categorical and continuous. For our purposes at this moment, we will just need a subset of the categorical emotions.



# Classifiers
## Haar-Cascade
I chose to use the Haar-Cascade Classifiers found in OpenCV to be able to reliably pull faces out of different images. The current focus was to make a model that could detect the emotion of the faces given however, further down the road Iwould like to revisit this to replace it with a Random Forest 

# CNN
![Example_CNN](Images/Face-Recognition-CNN-Architecture.png)

# Results
![Image](accuracy.png)