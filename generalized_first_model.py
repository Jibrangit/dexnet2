#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 07:44:15 2021

@author: mandar
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import os
DATA = os.getenv('DEXNET_DATA')
#print(DATA)   

##################################################################################################################

arrays = {}
# datapoint = '_05590.npz'                #Enter datapoint here (anywhere between _00000 and _06728, there are a total of 6.7 million datapoints i.e, 6728 x 1000). Change this value to just '.npz' to train on entire dataset
for filename in os.listdir(DATA):
    for npz_file_number in range (6500):   #Depending on number of examples just change the range.
        data = '_{0:05}.npz'.format(npz_file_number)
        #print(data)
        if filename.endswith(data):
            arrays[filename.replace(data, '')] = np.load(DATA + filename)
            #print(arrays)

features = {}
for array in arrays:
    f = arrays[array]
    feature = f['arr_0.npy']
    features[array] = feature
print(features.keys())

####################################################################################################################


# #Inputs to feed into CNN


aligned_imgs = features['depth_ims_tf']
# print(aligned_imgs.shape)
gripper_depths = features['hand_poses'][:, 2]          #aligned_imgs and gripper_depths are the "X" of our model (Refer to gqcnn/Data/README.md for more info)
# print(gripper_depths.shape)
grasp_metrics = features['robust_ferrari_canny']       #Y of our model (Grasp metric)

#Verify shapes 

# print(gripper_depths)
# print(grasp_metrics.shape)
# print(grasp_metrics)
# print(gripper_depths.shape)
# print(aligned_imgs.shape)



#******************One hot encoding***********************

# print(grasp_metrics[2])
def one_hot_encoding(grasp_metrics):
    for i in range(1000):
        if(grasp_metrics[i] > 0.002):  #threshold value is 0.002 readme
            grasp_metrics[i] = 1
        else:
            grasp_metrics[i] = 0
    return grasp_metrics

grasp_metrics = one_hot_encoding(grasp_metrics)

# print(grasp_metrics)



#######################################################################################################################

import tensorflow as tf
import math
import random

# example of defining the discriminator model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Lambda
from tensorflow.nn import local_response_normalization
from keras.utils.vis_utils import plot_model

from keras.layers.merge import concatenate
from keras.models import Model
# from tensorflow.keras import layers
from keras.layers import Dense, Input


#TODO: Initialize weights in layers according to the paper https://arxiv.org/pdf/1703.09312.pdf
def getGraspQualityVariable():
    input = Input(shape=(32, 32, 1), name="img")
    x1 = Conv2D(filters=64, kernel_size=7, activation='relu')(input)
    x2 = Conv2D(filters=64, kernel_size=5, activation='relu')(x1)
    x3 = Lambda(local_response_normalization)(x2)
    x4 = MaxPooling2D(pool_size=(2, 2), strides=2)(x3)
    x5 = Conv2D(filters=64, kernel_size=3, activation='relu')(x4)
    # x6 = Dropout(0.3)(x5)
    x6 = Conv2D(filters=64, kernel_size=3, activation='relu')(x5)
    x7 = Lambda(local_response_normalization)(x6)
    x8 = Flatten()(x7)
    x9 = Dense(1024, activation='relu')(x8)
    
    # plot_model(model, to_file='GraspQualityModel_plot.png', show_shapes=True, show_layer_names=True)
    return x9, input
    
getGraspQualityVariable()


def getPointcloudModel():
    input = Input(shape=(1), name="z")
    x = Dense(16, input_dim=1, activation='relu')(input)
    return x, input


from keras.layers.merge import concatenate
def getDexnet2Model():
    grasp_model, input_1 = getGraspQualityVariable()
    pc_model, input_2 = getPointcloudModel()

    out = Dense(1024, activation='relu')

    # x = layers.concatenate([grasp_model, pc_model])
    merge = concatenate([grasp_model, pc_model])
    out_1 = Dense(1024, activation='relu')(merge)
    out_2 = Dense(2, activation='softmax')(out_1)
    model = Model(inputs=[input_1, input_2], outputs=out_2)
    plot_model(model, to_file='Dexnet2_plot.png', show_shapes=True, show_layer_names=True)
    return model

first_model = getDexnet2Model()

first_model.summary()

##################################################################################################################

first_model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='/home/gqcnn_ws/src/first_model.weights.best.hdf5', verbose = 1, save_best_only=True)
x_train = [aligned_imgs, gripper_depths]                                   #Check model.summary() in previous section and check model architecture in paper
y_train = grasp_metrics                                                    #valued between [0, 1] grasp robustness for the given grasp
first_model.fit(x_train,
          y_train,
          batch_size=16,                                                   #worth ecperimenting
          epochs=10,                                      
          validation_split=0.25,                                            #Decide on a number
          callbacks=[checkpointer])        

                                #CHange the filepath of the checkpointer varianble to store different version of weights.


# first_model.predict(aligned_imgs[290])

