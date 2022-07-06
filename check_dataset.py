import numpy as np 
import os

DATA = os.getenv('DEXNET_DATA')

# x1_val_name = np.load('filename_strings/x1_val_filenames.npy')[2]
# x1_val = np.load(DATA+x1_val_name)['arr_0'][4:8]
# print(len(x1_val))

y_val_name = np.load('filename_strings/y_val_filenames.npy')
y_val = np.load(DATA + y_val_name[0])['arr_0']
print(y_val[0:20])
y = np.empty((16), dtype=int)     

y = np.load(DATA + y_val_name[1000])['arr_0'][300:316]