import os
import shutil
import numpy as np
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re 

from skimage.io import imread
from skimage.transform import resize

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

def SplitNames(x1_shuffled, x2_shuffled, y_shuffled):
    # Used this line as our filename array is not a numpy array.
    x1_shuffled_numpy = np.array(x1_shuffled)
    x2_shuffled_numpy = np.array(x2_shuffled)
    y_shuffled_numpy = np.array(y_shuffled)

    x1_train_filenames, x1_val_filenames = train_test_split(
        x1_shuffled_numpy, test_size=0.2, random_state=42)
    
    x2_train_filenames, x2_val_filenames = train_test_split(
        x2_shuffled_numpy, test_size=0.2, random_state=42)
    
    y_train_filenames, y_val_filenames = train_test_split(
        y_shuffled_numpy, test_size=0.2, random_state=42)

    print(x1_train_filenames.shape)
    print(x2_train_filenames.shape)
    print(y_train_filenames.shape) 

    print(x1_val_filenames.shape)
    print(x2_val_filenames.shape) 
    print(y_val_filenames.shape) 

    # You can save these files as well. As you will be using them later for training and validation of your model.
    np.save('filename_strings/x1_train_filenames.npy', x1_train_filenames)
    np.save('filename_strings/x2_train_filenames.npy', x2_train_filenames)
    np.save('filename_strings/y_train_filenames.npy', y_train_filenames)

    np.save('filename_strings/x1_val_filenames.npy', x1_val_filenames)
    np.save('filename_strings/x2_val_filenames.npy', x2_val_filenames)
    np.save('filename_strings/y_val_filenames.npy', y_val_filenames)

    print(len(x1_train_filenames), len(x1_val_filenames), len(y_train_filenames), len(y_val_filenames), len(x2_train_filenames), len(x2_val_filenames))
    # print(x1_val_filenames[700:710], y_val_filenames[700:710])             #Checking if splitting did not mix things up. 
    return x1_train_filenames, x1_val_filenames, x2_train_filenames, x2_val_filenames, y_train_filenames, y_val_filenames

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x1_IDs, x2_IDs, y_IDs, batch_size=32, dim1=(32,32, 1), dim2=1,
                 n_classes=2, shuffle=False):
        'Initialization'
        self.dim1 = dim1
        self.dim2 = dim2
        self.batch_size = batch_size
        self.x1_IDs = x1_IDs
        self.x2_IDs = x2_IDs
        self.y_IDs = y_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):                                                   #[DONE]
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))              
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]       

        # Generate data
        x1, x2, y = self.__data_generation(indexes)         

        return [x1, x2], y                                                #This is the function that returns data when it is required while training. 
    
    def on_epoch_end(self):                                               #Shuffling already done, so dont worry about this
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x1 = np.empty((self.batch_size, *self.dim1))
        x2 = np.empty((self.batch_size, *self.dim2))
        y = np.empty((self.batch_size), dtype=float)

        thousands = indexes//1000
        ones = indexes - thousands 

        # Generate data   
        # Store sample
        x1[i,] = np.load('DEXNET_DATA' + self.x1_IDs[thousands])['arr_0.npy'][ones]
        x2[i,] = np.load('DEXNET_DATA' + self.x2_IDs[thousands])['arr_0.npy'][ones][2]
        y[i,] = np.load('DEXNET_DATA' + self.y_IDs[thousands])['arr_0.npy'][ones]


        return x1, x2, y


if __name__ == '__main__':
    x1, x2, y = NameStore()
    x1_shuffled, x2_shuffled, y_shuffled = StoreNames(x1, x2, y)
    x1_train_filenames, x1_val_filenames, x2_train_filenames, x2_val_filenames, y_train_filenames, y_val_filenames = SplitNames(x1_shuffled, x2_shuffled, y_shuffled)
