from third_model import getDexnet2Model, getPointcloudModel, getGraspQualityVariable
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

if __name__ == '__main__':
    DATA = os.getenv('DEXNET_DATA')
    third_model_pred = getDexnet2Model()
    print(third_model_pred.summary())
    third_model_pred.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    third_model_pred.load_weights('weights/third_model.weights.best.hdf5')

    random_data_thousands = np.random.randint(0, 6727)
    random_data_ones = np.random.randint(0, 1000)

    print(f'Loading datapoint {(random_data_thousands * 1000) + random_data_ones}')

    img_x1 = np.load(DATA + 'depth_ims_tf_table_0' + str(random_data_thousands)+str('.npz'))['arr_0'][random_data_ones]
    z_x2 = np.load(DATA + 'hand_poses_0' + str(random_data_thousands)+str('.npz'))['arr_0'][random_data_ones][2]
    metric_y = np.load(DATA + 'robust_ferrari_canny_0' + str(random_data_thousands)+str('.npz'))['arr_0'][random_data_ones]

    third_model_pred.predict([img_x1, z_x2])
