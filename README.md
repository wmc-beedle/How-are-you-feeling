# How are you feeling?
---
Guessing what emotion someone is feeling is usually pretty easy for a human to do. How would you feel about guessing at 100?1000?1,000,000? different faces? I am willing to bet that even at those large numbers, most people would be fairly accurate but there is a catch; going through that many faces would most likely take up a lot of time. Even with that caveat though, that information is still very valuable so we are put in a position where machine learning excels: large data crunching to find patterns.

# Goal

Through using machine learning algorithm, in this case specifically a convolutional neural network(ConvNet/CNN), my goal is to make a model that when fed an image, it will be able to seek out the face within that image and return to us a prediction of what that person may be experiencing at the moment. It's uses would primarily be in marketing and scientific research to study more human behavior analysis.


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

To use this particular model you will need to be using the EMOTIC Dataset provided here[http://sunai.uoc.edu/emotic/]. The data provided here can be split into to type: categorical and continuous. This is important particularly for us as using a CNN you will need to deal with difference before inputing it into the CNN. The categorical data are predicted emotion(s), Gender, and Age. While the other two may not be included in this initial model, they are still valuable sources of data that can be used in a future iteration of this model. The cremaining data are BBox(Boundry boxes for where the face in the image is), Negative/Positive, Calm, Active, Dominated/ In Control. As with the Gender and Age data, besides the BB data, the remainder is still valuable and will be put to the side for now. This dataset furthermore already comes in three parts: training, testing, and valuation sets. Each file has 9 columns with the training set having 23,265 rows, testing having 7,202 rows, and validation having 3,314 rows. 


# CNN
![Example_CNN](Images/Face-Recognition-CNN-Architecture.png)

*The image above is purely for explanation purposes of how the model works and not representative of the model itself.*
 For this model, we will be using the image, BBox, and Categorical Labels data to train our CNN. Before
 
# Instructions

*The file used to preprocess the annotations from mat to csv file will be added back on once recovered.*
The emotic dataset will be needed to be downloaded into the "Data" file. From there, it will also be necessary to import the [Haar Cascade Classifiers](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) into the primary folder. At this point, import to emotic dataset as is into the Data/emotic folder. Running mat_to_csv.py will convert the file into enother file in the directory called emotic_pre(pre-processing). Within this folder are the three datasets sorted proper for the model to run. Follow up within running image_pre to pre-train the Haar-Cascade Classifiers. From here, running model_maker.py should find all the pre-processed images and produce a CNN model! *In the code's current structure, the model is unable to be created. Once the mat_to_csv.py file and the model_maker



# Results
![sad](Images/face_emotion.png)
The model for this image had a 48% accuracy and log loss of 1.4. It was run on a subset of the full range of emotions in an attempt to increase accuracy. 

# Next Steps
Follow-up steps would be:
to make the model functional on all emotions
make functional with a webcam
