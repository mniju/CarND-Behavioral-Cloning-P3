import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import os
import cv2
# Utilities to split the training set
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Keras utilities
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.activations import relu, softmax, elu
from keras.optimizers import Adam
from keras import backend as K
#from keras.models import load_model
import h5py #For Saving the Model

'''Read the Data'''
csvdata = pd.read_csv('driving_log.csv')
# Get no of samples
nTrain = csvdata.shape[0]

'''Split the Data'''
#Shuffle the Data
data_shuffle = shuffle(csvdata)
#Split Training and Test Data
train_data,test_data = train_test_split(data_shuffle, test_size = 0.2)

'''Remove the ZERO steering angle to reduce the steering bias towards ZERO'''
# Remove Zero Steering Angle
idx = train_data['steering']!=0
train_data = train_data[idx].reset_index()
# Get the Individual Data
L_Images,C_Images,R_Images,Steerings = train_data['left'],train_data['center'],train_data['right'],train_data['steering']

#AUGMENTATION
#1.Flip the Image
# https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.j70p8njh9
def flip_img(image,steer_angle):
    #Flip image at random
    #new_image = image
    #new_angle = steer_angle
    #if np.random.randint(2) == 0:
    #new_image = np.fliplr(image)
    #new_angle = -new_angle
    return np.fliplr(image),-steer_angle
	
#2.Random Left and right Shift
def trans_image(image,steer_angle,trans_range):
    # Translation
    num_rows, num_cols = image.shape[:2]
    #Apply random shifts in horizontal direction of upto *10* pixels, and apply angle change of *.2* per pixel.
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer_angle + tr_x/trans_range*2*.2
    #steer_angle_per_pixel = 1/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #Translation Matrix M150
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # Execute the Translation
    image_T = cv2.warpAffine(image,Trans_M,(num_cols,num_rows))
    return image_T,steer_ang

#3.Brightness
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.qexmmzhpe
def augment_brightness(image):
    #Convert to HUse saturation Value to change the brightness
    imgB = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(.25,1.25)
    #random_bright = .25+np.random.uniform()
    # Slice the V channel and chnage the brightness
    imgB[:,:,2] = imgB[:,:,2]*random_bright
    imgB = cv2.cvtColor(imgB,cv2.COLOR_HSV2RGB)
    return imgB
	
'''Resize and Crop Image for the Nvidia Model'''
def img_resize(image,new_size):
    return cv2.resize(image,(new_size[0],new_size[1]), interpolation=cv2.INTER_AREA)

def img_crop(image,y_crop):
    # img[y:y+h, x:x+w] -> 35 from top until 130 in bottom
    image = image[y_crop[0]:y_crop[1], 0:image.shape[1]]
    return image
def int_process(imagepath,y_crop,new_size):
    image = plt.imread(imagepath)
    image = img_crop(image,y_crop)
    image = img_resize(image,new_size)
    return image
	
'''Function to Get a Batch of data with Augmentation'''
def images_per_line(batch_data):
    batch_data = shuffle(batch_data)
    L_Images,C_Images,R_Images,Steerings = batch_data['left'],batch_data['center'],batch_data['right'],batch_data['steering']
    batch_imgs = np.empty((0,66,200,3),dtype= np.uint8)
    batch_steers = np.array([])
    aug_imgs = np.empty((0,66,200,3),dtype= np.uint8)
    aug_steers = np.array([])
    y_crop = (50,140)
    new_size = (200,66)
    
    for L_Image,C_Image,R_Image,Steer in zip(L_Images,C_Images,R_Images,Steerings):
        imgs = np.empty((0,66,200,3),dtype= np.uint8)
        steers = np.array([])
        aug_imgs = np.empty((0,66,200,3),dtype= np.uint8)
        aug_steers = np.array([])
        
        #Just add offset to steering angle for the right and left image
        l_img = int_process(L_Image.strip(),y_crop,new_size)
        steer_l = Steer + 0.25
        print(steer_l)
        imgs = np.append(imgs,[l_img],axis =0)
        steers = np.append(steers,steer_l)
        # Now right Image
        r_img = int_process(R_Image.strip(),y_crop,new_size)
        steer_r = Steer - 0.25
        imgs = np.append(imgs,[r_img],axis =0)
        steers = np.append(steers,steer_r)
        #Now centre image - No Change
        c_img = int_process(C_Image.strip(),y_crop,new_size)
        steer_c = Steer
        imgs = np.append(imgs,[c_img],axis =0)
        steers = np.append(steers,steer_c)

        for img,steer in zip(imgs,steers):
            #Flip the Image
            flipped_img,flip_steer = flip_img(img,steer)
            aug_imgs = np.append(aug_imgs,[flipped_img],axis =0)
            aug_steers = np.append(aug_steers,flip_steer)
            #Brightness
            bright_img = augment_brightness(img)
            aug_imgs = np.append(aug_imgs,[bright_img],axis =0)
            aug_steers = np.append(aug_steers,steer)
            # Translate
            trans_img,trans_steer =  trans_image(img,steer,trans_range=100)
            aug_imgs = np.append(aug_imgs,[trans_img],axis =0)
            aug_steers = np.append(aug_steers,trans_steer)
        # Append the images with the Augmented images    
        imgs = np.append(imgs,aug_imgs,axis =0)
        steers = np.append(steers,aug_steers,axis =0)
        batch_imgs = np.append(batch_imgs,imgs,axis =0)
        batch_steers = np.append(batch_steers,steers,axis =0)
        return batch_imgs,batch_steers
'''Generator for a Batch Data'''
def image_batch_generator(data,batch_size=16):
    while 1:
        for offset in range(0,data.shape[0],batch_size):
            print('start:',offset,';','end:',offset+batch_size)
            batch_data = data[offset:offset+batch_size]
            batch_imgs,batch_steers = images_per_line(batch_data)
            yield batch_imgs,batch_steers*1.05 # A small gain to correct in curves
# Assign the Generator
train_generator = image_batch_generator(train_data,batch_size=16)
validation_generator = image_batch_generator(train_data,batch_size=16)

'''The Nivdia Model'''
model = Sequential()
#(1) Input 3 @ 66 x 200
ch, row, col = 3, 66, 200  # Resized Format
# Channel Ordering for TF and Keras
# http://stackoverflow.com/questions/39815518/keras-maxpooling2d-layer-gives-valueerror
# Cropping [Already Completed]
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#(2) Normlaizing Layer
model.add(Lambda(lambda x: x/127.5 - 1,input_shape=(row, col, ch),output_shape=(row, col, ch)))
#(3) Convolution Layer -1
#	Input: 3@ 66 x 200 | Output: 24@ 31 x 98 | strides 2x2 | kernel:5x5 
model.add(Convolution2D(24, 5, 5,subsample=(2, 2), border_mode='valid',init = 'he_normal'))
model.add(Activation('elu'))
#(4) Convolution Layer -2
#	Input: 24@ 31 x 98# |Output: 36@ 14 x 7 | strides 2x2 | kernel:5x5 
model.add(Convolution2D(36, 5, 5,subsample=(2, 2), border_mode='valid',init = 'he_normal'))
model.add(Activation('elu'))
#(4) Convolution Layer -3
#	Input: 36@ 14 x 7 | Output: 48@ 5 x 22 | strides 2x2 | kernel:5x5 
model.add(Convolution2D(48, 5, 5,subsample=(2, 2), border_mode='valid',init = 'he_normal'))
model.add(Activation('elu'))
#(5) Convolution Layer -4
#	Input: 48@ 5 x 22 | Output: 64@ 3 x 20 | strides 1x1 | kernel:3x3 
#   Apply: 3x3 convolution
model.add(Convolution2D(64, 3, 3,subsample=(1, 1), border_mode='valid',init = 'he_normal'))
model.add(Activation('elu'))
#(6) Convolution Layer -5
#	Input: 64@ 3 x 20 | Output: 64@ 1 x 18 | strides 1x1 | kernel:3x3 
#   Apply: 3x3 convolution
model.add(Convolution2D(64, 3, 3,subsample=(1, 1), border_mode='valid',init = 'he_normal'))
model.add(Activation('elu'))
#(7) Flatten 
#	Input: 64@ 1 x 18 | Output: 1164 Neurons
model.add(Flatten(input_shape=(1, 18,64)))
#(7) Fully Connected 
#	Input: 1164 Neurons | Output: 100 Neurons
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(.5))
#(8) Fully Connected 
#	Input: 100 Neurons | Output: 50 Neurons
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(.5))
#(8) Fully Connected ,Input: 50 Neurons | Output: 10 Neurons 
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dropout(.5))
#(8) Fully Connected 
#	Input: 10 Neurons | Output: 1 Neuron
model.add(Dense(1))
model.add(Activation('linear'))

# Compile , Run and save the Model
epochs = np.array([5,7,10,13,15]) # Epochs to run and save
model.compile(loss='mse', optimizer='adam')
for epoch in epochs:
    model.fit_generator(train_generator, samples_per_epoch=
                    train_data.shape[0], validation_data=validation_generator,
                    nb_val_samples=test_data.shape[0], nb_epoch=epoch)
    modelpath = 'gain1_08\model_epoch_'+ str(epoch) + '.h5'
    model.save(modelpath)  # creates a HDF5 file for each epoch in epochs
print('Completed')