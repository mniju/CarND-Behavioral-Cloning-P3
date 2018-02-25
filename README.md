# **Behavioral Cloning** 

## Writeup for Project Submission

**Behavrioal Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image3]: ./writeup/randombrightness.png "Brigntness Image"
[image4]: ./writeup/randomTranslation.png "randomTranslation Image"
[image7]: ./writeup/flipped.png "Flipped Image"
[image8]: ./writeup/cnn-architecture-624x890.png "Architecture Image"
[image9]: ./writeup/original.png "Sample Image"
[image10]: ./writeup/CroppingandResizing.png "cropping&Resizing Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** summarizing the results
* **video.mp4** to visualize Autonomous driving for a lap.

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```
I tested the code in anaconda windows enviornment .
#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training, the augmentation i added for training the network and validating the model. It contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

I used the NVDIA End to End Deep Learning Model as proposed in this [paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

The  model consists of a 5 convolution neural network with filter sizes 5x5 and 3x3, and depths between 24 and 64 (model.py lines 170-188) and 4 Fully connected layer (model.py lines 195-210)

The model includes ELU layers to introduce nonlinearity after the convolution layers (code line 171..), and the data is normalized in the model using a Keras lambda layer (code line 167).

Additionally i used a keras lambda layer to Normalize the images between -1 and 1



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers (a bit aggressively)  in order to reduce overfitting (model.py lines 197,202,206).
I used a dropout of 50 % in all the Fully connected layers so that the network dont form too much memory of the track and try to over fit. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 215-218). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 214).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

I used the Udacity training data.I found the training data from Udacity had more data for steering angle zero.(Driving straight). I went ahead and removed the straight steering data (Zero) to avoid the neural network developing a bias towards driving straight.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an existing  standard model as base and further tune on that model to make the car drive on the track autonomously.

My initial search for an existing base model ended up with two models.

* comma ai [model](https://github.com/commaai/research/blob/master/train_steering_model.py)
* Nvdia [Model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

I thought of starting with Nvdia, check how things work and then if it fails then stick to Comma ai. Luckily my car stayed on track with Nvdia model 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I found that my first test in the track resulted in more wobbling of the car towards left and right and it doesn't take turns properly and goes outside the track.

To fix this i analysed the data and found that more data is with steering angle '0'. I removed the data with steering angle '0' to remove the bias towards driving straight.

After removing the zero steering angle data,when i tested model in the track,the car was driving but wobbling relentlesly. Perhaps it over fitted to take turns.

Time to add more dropouts to prevent overfitting. Initially i added 20 % , 30% and 50 % dropouts in the Fully connected layer1,2 and 3 respectively.The wobbling reduced. However , the car isn't driving steadily. Finally i made a droput of 50 % for all the fully connected layers to avoid any overfitting.(model.py lines 192,201 and 207)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The  model consists of a 5 convolution neural network with filter sizes 5x5 and 3x3, and depths between 24 and 64 (model.py lines 170-188) and 4 Fully connected layer (model.py lines 195-210)

The model includes ELU layers to introduce nonlinearity after the convolution layers (code line 171..), and the data is normalized in the model using a Keras lambda layer (code line 167).

Here is a visualization of the architecture 


![alt text][image8]

#### 3. Creation of the Training Set & Training Process

I started with the Udacity provided data set.

![alt text][image9]

I used the right and left camera images to calculate for recovery by adding the offset in the steering angle for each image as mentioned in the Nvidia model.

The image has to be cropped to avoid learning items other than the road  and resized as per the model requirement.

Original Image    --->   Top and Bottom Cropped    --->    Reshaped Image

![alt text][image10]

##### Augmentation:

To augment the data set, I  

* Flipped images(model.py lines 43-50) ![alt text][image7]
* Translated the image to generate random steering(model.py lines 53-65)![alt text][image4]
* Change brightness randomly(model.py lines 69-77)![alt text][image3]


I used python generator to generate the augmented data for each data on the fly to minimize the memory cost(model.py lines 147-153)

I randomly shuffled the data set(before augmentation) and put 20% of the data into a validation set. (model.py lines 29 and 31)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.But this was pretty insignificant as we can check  the car in the actual track itself. There is no ideal number of epochs as such. I got good results at after training 5-7 epochs . I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Conclusion:
This project was really challenging and i enjoyed working in it.In future i plan  generalizing it for Track 2 too.