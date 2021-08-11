# COVID19_RT_FaceMaskDetector
This project provides the ability to detect whether a person wearing a mask or not, by using Convolutional Neural Network (CNN)

Libraries & Frameworks: Keras, TensorFlow, MobileNetV2 architecture, openCV

This project contains 2 main phases: 

Phase 1: Training (code1.py) 

1. Initialize the initial learning rate, number of epochs to train for,and batch size.

2. Data Preprocessing -  load the input image (224x224) and preprocess it, update the data and labels lists, respectively.

3. Partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing.

4. Load the MobileNetV2 network, ensuring the head FC layer sets are left off.

5. Construct the head of the model that will be placed on top of the base model.

6. Place the head FC model on top of the base model (this will become the actual model we will train).

7. Loop over all layers in the base model and freeze them so they will not be updated during the first training process.

8. Compile the model and train the head of the network.

9. For each image in the testing set, we need to find the index of the label with corresponding largest predicted probability.

Phase 2: Deployment (code2.py)

1. Grab the dimensions of the frame and then construct a blob from it and pass the blob through the network and obtain the face detections.

2. Initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network.

3. Extract the confidence (i.e., probability) associated with the detection and filter out weak detections by ensuring the confidence is greater than the minimum confidence.

4. Only make a predictions if at least one face was detected.

5. Load our serialized face detector model and load the face mask detector model.

6. Loop over the frames from the video stream.

7. Loop over the detected face locations and their corresponding locations.

8. Display the label and bounding box rectangle on the output frame.

Results: 


![image](https://user-images.githubusercontent.com/54996146/129073783-cc088c29-cee0-461c-aa74-783b4787347a.png)


![113990852-90dfd980-985a-11eb-8ae0-10fcbfeeb316](https://user-images.githubusercontent.com/54996146/129074226-74f04445-05e3-4a3f-9a70-719456214d2d.gif)
