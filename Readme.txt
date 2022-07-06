To run training:

1. Download the dataset from https://berkeley.app.box.com/s/6mnb2bzi5zfa7qpwyn7uq5atb7vbztng/folder/25803680060 
2. Set an environment variable to the dataset called 'DEXNET_DATA' or alternatively change the 
DATA variable in third_model.py to the dataset path. 

3. Run the 'check_dataset.py' to check if everything is currently loaded, and also do 'rm -rf *6728 * in the terminal for deleting the last datapoint as it contains 850 files and causes problems in the training.

4. Set the NUM_EXAMPLES to anywhere between 1 and 6726 in the 'main' in third_model.py and run the file to start training.


