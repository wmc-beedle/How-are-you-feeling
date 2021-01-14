# How are you feeling?
---
Discovering what emotion someone is feeling is ususally

# Goal

Make a model that when fed an image can find a face and estimate as to what emotion the individual is experiencing. It's uses would primarily be in marketing and scientific research to help retrieve more geniuine reactions from a large group of people.


# Dependencies

* Pandas
* Numpy
* Sklearn
* OpenCV 
* Tensorflow
* tqdm
* ast

# [Emotic Dataset](http://sunai.uoc.edu/emotic/)

![Emotic_logo](Images/emotic.png)


*The EMOTIC dataset, named after EMOTions In Context, is a database of images with people in real environments, annotated with their apparent emotions. The images are annotated with an extended list of 26 emotion categories combined with the three common continuous dimensions Valence, Arousal and Dominance.*

*R. Kosti, J.M. Álvarez, A. Recasens and A. Lapedriza, "Context based emotion recognition using emotic dataset", IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2019.*

This dataset already comes in three parts: training, testing, and valuation sets. Each file has 9 columns with the training set having 23,265 rows, testing having 7,202 rows, and validation having 3,314 rows. 


# [Haar-Cascade Classifiers](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
Created by Paul Viola and Michael Jones and uses Haar kernels for extracing features and it uses multiple classifiers one after an other. The classifiers get more complex as data moves forward. Each classifier will specify whether the image is maybe from the desired class or it is definitly not in the desired class and if it is maybe from the desired class will pass image forward to the next classifier.

# Instructions

*The file used to preprocess the annotations from mat to csv file will be added back on once recovered.*
The emotic dataset will be needed to be downloaded into the "Data" file. From there, it will also be necessary to import the [Haar Cascade Classifiers](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) into the primary folder. At this point, import to emotic dataset as is into the Data/emotic folder. Running mat_to_csv.py will convert the file into enother file in the directory called emotic_pre(pre-processing). Within this folder are the three datasets sorted proper for the model to run. Follow up within running image_pre to pre-train the Haar-Cascade Classifiers. From here, running model_maker.py should find all the pre-processed images and produce a CNN model! 


# CNN
![Example_CNN](Images/Face-Recognition-CNN-Architecture.png)

*The image above is purely for explanation purposes of how the model works and not representative of the model itself.*



# Results
![sad](Images/face_emotion.png)
The model for this image had a 48% accuracy and log loss of 1.4. It was run on a subset of the full range of emotions in an attempt to increase accuracy. 

# Next Steps
Follow-up steps would be:
to make the model functional on all emotions
make functional with a webcam
