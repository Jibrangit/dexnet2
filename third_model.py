import os
import shutil
import numpy as np
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re 
import time

from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
import matplotlib.pyplot as plt
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
from keras.utils import to_categorical

from keras.layers.merge import concatenate
from keras.models import Model
# from tensorflow.keras import layers
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def NameStore():
    data_dir = os.getenv('DEXNET_DATA')
    subdirs, dirs, files = os.walk(data_dir).__next__()   #Iterate through all images and stores their names in 'files'.
    m = len(files)
    # print(m//3)

    x1_names = []
    x2_names = []
    y_names  = []

    for subdir, dirs, files in os.walk(data_dir):         #Store those filenames in 3 different list, two Xs and Y.
        #print(files)
        for file in files:
            if("depth" in file):
                x1_names.append(file)
            if("hand" in file):
                x2_names.append(file)
            if("robust" in file):
                y_names.append(file)
    
    x1_names.sort(key=natural_keys)
    x2_names.sort(key=natural_keys)
    y_names.sort(key=natural_keys)

    # print(len(x1_names), len(x2_names), len(x1_names))   #Check if all dataset has been loaded.
    # print(y_names)                                       #To check if strings have been sorted.
    return x1_names, x2_names, y_names

def StoreNames(x1, x2, y):
    # saving the filename array as .npy file
    np.save('filename_strings/x1_names.npy', x1)
    np.save('filename_strings/x2_names.npy', x2)
    np.save('filename_strings/y_names.npy', y)

    x1_shuffled, x2_shuffled, y_shuffled = shuffle(x1, x2, y)
    np.save('filename_strings/x1_shuffled_names.npy', x1_shuffled)
    np.save('filename_strings/x2_shuffled_names.npy', x2_shuffled)
    np.save('filename_strings/y_shuffled_names.npy', y_shuffled)

    # print(x1_shuffled[300:310])                          #Checking if shuffling has been consistent across datapoints.
    # print(x2_shuffled[300:310])
    # print(y_shuffled[300:310])

    return x1_shuffled, x2_shuffled, y_shuffled

def SplitNames(x1_shuffled, x2_shuffled, y_shuffled, NUM_EXAMPLES):
    # Used this line as our filename array is not a numpy array.
    x1_shuffled_numpy = np.array(x1_shuffled)[0:NUM_EXAMPLES]
    x2_shuffled_numpy = np.array(x2_shuffled)[0:NUM_EXAMPLES]
    y_shuffled_numpy = np.array(y_shuffled)[0:NUM_EXAMPLES]

    x1_train_filenames, x1_val_filenames = train_test_split(
        x1_shuffled_numpy, test_size=0.2, random_state=42)
    
    x2_train_filenames, x2_val_filenames = train_test_split(
        x2_shuffled_numpy, test_size=0.2, random_state=42)
    
    y_train_filenames, y_val_filenames = train_test_split(
        y_shuffled_numpy, test_size=0.2, random_state=42)

    # print(x1_train_filenames.shape)                                       #Checking them for split-correctness
    # print(x2_train_filenames.shape)
    # print(y_train_filenames.shape) 
    # print(x1_val_filenames.shape)
    # print(x2_val_filenames.shape) 
    # print(y_val_filenames.shape) 

    # You can save these files as well. As you will be using them later for training and validation of your model.
    np.save('filename_strings/x1_train_filenames.npy', x1_train_filenames)
    np.save('filename_strings/x2_train_filenames.npy', x2_train_filenames)
    np.save('filename_strings/y_train_filenames.npy', y_train_filenames)

    np.save('filename_strings/x1_val_filenames.npy', x1_val_filenames)
    np.save('filename_strings/x2_val_filenames.npy', x2_val_filenames)
    np.save('filename_strings/y_val_filenames.npy', y_val_filenames)

    # print(len(x1_train_filenames), len(x1_val_filenames), len(y_train_filenames), len(y_val_filenames), len(x2_train_filenames), len(x2_val_filenames)) 
    # print(x1_val_filenames[700:710], y_val_filenames[700:710])             #Checking if splitting did not mix things up. 
    return x1_train_filenames, x1_val_filenames, x2_train_filenames, x2_val_filenames, y_train_filenames, y_val_filenames

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_type, x1_IDs, x2_IDs, y_IDs, batch_size=16, dim1=(32, 32, 1),
                 n_classes=2, shuffle=False):
        'Initialization'
        self.dataset_type = dataset_type
        self.dim1 = dim1
        self.batch_size = batch_size
        self.x1_IDs = x1_IDs
        self.x2_IDs = x2_IDs
        self.y_IDs = y_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.classes = n_classes
        

    def __len__(self):                                               
        'Denotes the number of batches per epoch'
        num_batches =  int(np.floor((len(self.x1_IDs) * 1000) / self.batch_size))      
        # print(f'No of batches for dataset of shape {self.x1_IDs.shape}= {num_batches}')                                  #Checking to see if number of batches is correct    
        return num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # print(f'In get_item, Index No: {index}')
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]   
        # print(f'Some indices values: {self.indexes[0:20]}')               #Looks correct. 
        # print(f'Indexes being retrieved: {indexes}')    

        # Generate data
        x1, x2, y = self.__data_generation(indexes)  
        # print(f'Shape of x1:{x1.shape}, x2:{x2.shape}, y:{y.shape}')     #Shape of x1 as of now is [16, 32, 32, 1] , x2=16, y=16

        return [x1, x2], y                                                 #This is the function that returns data when it is required while training. 
    
    def on_epoch_end(self):                                                #Shuffling already done, so dont worry about this
        'Updates indexes after each epoch'
        # print('End of epoch')
        self.indexes = np.arange(len(self.x1_IDs) * 1000, dtype=int)
        # print(f'Number of indices = {self.indexes}')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'                      # X : (n_samples, *dim, n_channels)
        # Initialization
        # print("In Data generator")
        x1 = np.empty((self.batch_size, *self.dim1))
        x2 = np.empty((self.batch_size), dtype=float)
        y = np.empty((self.batch_size), dtype=int)                          #Watch out for the type casting 
        DATA = os.getenv('DEXNET_DATA')

        thousands = indexes // 1000
        ones = indexes - (thousands*1000)

        x1_name = self.x1_IDs[thousands]
        x2_name = self.x2_IDs[thousands]
        y_name = self.y_IDs[thousands]
        

        for i in range(len(indexes)):
            x1[i] = np.load(DATA + x1_name[i])['arr_0'][ones[i]]
            x2[i] = np.load(DATA + x2_name[i])['arr_0'][ones[i]][2]
            y[i] = np.load(DATA + y_name[i])['arr_0'][ones[i]]
        # if(self.dataset_type == "Validation"):
        y = self.one_hot_encoding(y)
        # print(f'\nFiles being called for {self.dataset_type} dataset are x1: {x1_name} with arrays {ones}, x2: {x2_name} and y: {y}')


        # print(f'x1 size is {len(x1)}')
        return x1, x2, y
    
    def one_hot_encoding(self, grasp_metrics):
        for i in range(self.batch_size):
            if(grasp_metrics[i] > 0.002):  #threshold value is 0.002 readme
                grasp_metrics[i] = 1
            else:
                grasp_metrics[i] = 0
        return grasp_metrics

def getGraspQualityVariable():
    input = Input(shape=(32, 32, 1), name="img")
    x1 = Conv2D(filters=64, kernel_size=7, activation='relu')(input)
    x2 = Conv2D(filters=64, kernel_size=5, activation='relu')(x1)
    x3 = Lambda(local_response_normalization)(x2)
    x4 = MaxPooling2D(pool_size=(2, 2), strides=2)(x3)
    x5 = Conv2D(filters=64, kernel_size=3, activation='relu')(x4)
    x6 = Conv2D(filters=64, kernel_size=3, activation='relu')(x5)
    x7 = Lambda(local_response_normalization)(x6)
    x8 = Flatten()(x7)
    x9 = Dense(1024, activation='relu')(x8)
    
    # plot_model(model, to_file='GraspQualityModel_plot.png', show_shapes=True, show_layer_names=True)
    return x9, input

def getPointcloudModel():
    input = Input(shape=(1), name="z")
    x = Dense(16, input_dim=1, activation='relu')(input)
    return x, input

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

if __name__ == '__main__':
    NUM_EXAMPLES = 6726
    x1, x2, y = NameStore()                                                 
    x1_shuffled, x2_shuffled, y_shuffled = StoreNames(x1, x2, y)
    x1_train_filenames, x1_val_filenames, x2_train_filenames, x2_val_filenames, y_train_filenames, y_val_filenames = SplitNames(x1_shuffled, x2_shuffled, y_shuffled, NUM_EXAMPLES)
    
    #=========================================================================================================================#

    # x1_train_filenames = np.load('filename_strings/x1_train_filenames.npy')
    # x2_train_filenames = np.load('filename_strings/x2_train_filenames.npy')
    # y_train_filenames = np.load('filename_strings/y_train_filenames.npy')
    # x1_val_filenames = np.load('filename_strings/x1_val_filenames.npy')
    # x2_val_filenames = np.load('filename_strings/x2_val_filenames.npy')
    # y_val_filenames = np.load('filename_strings/y_val_filenames.npy')

    third_model = getDexnet2Model()
    print(third_model.summary())
    third_model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    training_generator = DataGenerator(dataset_type="Training", x1_IDs=x1_train_filenames, x2_IDs=x2_train_filenames, y_IDs=y_train_filenames)
    validation_generator = DataGenerator(dataset_type="Validation", x1_IDs=x1_val_filenames, x2_IDs=x2_val_filenames, y_IDs=y_val_filenames)

    checkpointer = ModelCheckpoint(filepath='weights/third_model.weights.best.hdf5', verbose = 1, save_best_only=True)
    third_model.save_weights('weights/third_model.weights.best.hdf5')
    start_time = time.time()
    
    third_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    shuffle=False
                    )

    end_time = time.time() - start_time
    print(f'Time elapsed for {NUM_EXAMPLES * 1000} datapoints = {end_time} seconds, {end_time/60} minutes, {end_time/3600} hours')
