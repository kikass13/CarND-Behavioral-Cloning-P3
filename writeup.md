# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model.h5 run_kik
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the nvidia model which is structured like this:

```
Layer (type)                     Output Shape          Param #     Connected to   
==================================================================================
cropping2d_3 (Cropping2D)        (None, 65, 320, 3)    0         cropping2d_input_3[0][0]         
__________________________________________________________________________________________
lambda_3 (Lambda)                (None, 65, 320, 3)    0           cropping2d_3[0][0]               
__________________________________________________________________________________________
convolution2d_11 (Convolution2D) (None, 33, 160, 24)   1824        lambda_3[0][0]                   
__________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 17, 80, 36)    21636       convolution2d_11[0][0]           
__________________________________________________________________________________________
convolution2d_13 (Convolution2D) (None, 9, 40, 48)     43248       convolution2d_12[0][0]           
__________________________________________________________________________________________
convolution2d_14 (Convolution2D) (None, 9, 40, 64)     27712       convolution2d_13[0][0]           
__________________________________________________________________________________________
convolution2d_15 (Convolution2D) (None, 9, 40, 64)     36928       convolution2d_14[0][0]           
__________________________________________________________________________________________
flatten_3 (Flatten)              (None, 23040)         0           convolution2d_15[0][0]           
__________________________________________________________________________________________
dense_9 (Dense)                  (None, 100)           2304100     flatten_3[0][0]                  
__________________________________________________________________________________
dropout_5 (Dropout)              (None, 100)           0           dense_9[0][0]                    
__________________________________________________________________________________
activation_5 (Activation)        (None, 100)           0           dropout_5[0][0]                  
__________________________________________________________________________________
dense_10 (Dense)                 (None, 50)            5050        activation_5[0][0]               
__________________________________________________________________________________
dropout_6 (Dropout)              (None, 50)            0           dense_10[0][0]                   
__________________________________________________________________________________________
activation_6 (Activation)        (None, 50)            0           dropout_6[0][0]                  
__________________________________________________________________________________________
dense_11 (Dense)                 (None, 10)            510         activation_6[0][0]               
__________________________________________________________________________________________
dense_12 (Dense)                 (None, 1)             11          dense_11[0][0]                   
==================================================================================
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the left and right images and added an angular offset to force the vehicle into the middle.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network from nvidia. I thought this model might be appropriate because is uses multiple convolutional layers for image abstraction. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Also the steering angle looked kind of weak in the previously given training set (seen in histogram), so i just altered it slightly to give the net a strong center oriented goal. Two Dropout layers (a hard one with 50% after the convolutions and a soft one 25% in the middle of the flattened layers) prevent overfitting of the data.

Then I played around with hyperparameters and different image configurations (center, left and right, all of them etc.).

Every iteration of my network was tested in the simulator, the results helped me finding a compromise between training time and successful driving. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, which was recorded in a video.

#### 3. Creation of the Training Set & Training Process

As learned in previous projects, i used grayscale normalized images and split up my dataset into a training and a validation set. These were shuffled to ensure the training would occur more "natural". I recorded some rounds and recoveries myself, but quickly learned that these were not needed and (sometimes) in fact destroyed the functionality of my network (because of the reduced training time needed on my cpu).

To augment the data set, I also flipped images and angles, which would lead to more data and more abstract information for the network.
