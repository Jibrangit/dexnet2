import numpy as np
import os
import matplotlib.pyplot as plt
arrays = {}
datapoint = '_05590.npz'                #Enter datapoint here (anywhere between _00000 and _06728)
for filename in os.listdir('/home/jibran/dexnet_ws/gqcnn/data/training/dex-net_2.0/tensors'):
    if filename.endswith(datapoint):
        arrays[filename.replace(datapoint, '')] = np.load('/home/jibran/dexnet_ws/gqcnn/data/training/dex-net_2.0/tensors/'+filename)
#Visualization
index = np.random.randint(0, 1000)
features = {}
print('Index = ', index)
for array in arrays:
    f = arrays[array]
    feature = f['arr_0.npy']
    features[array] = feature
    if(len(feature.shape) > 2):
        plt.imshow(feature[index, :, :, :])
        plt.title(array+ ": "+ "Index "+str(index))
        # plt.show()
    else:
        print(array, ": ", feature[index])
#Inputs to feed into CNN
print(features.keys())
aligned_imgs = features['depth_ims_tf_table']
# print(aligned_imgs.shape)
gripper_depths = features['hand_poses'][:, 2]          #aligned_imgs and gripper_depths are the X of our model.
# print(gripper_depths.shape)
grasp_metrics = features['robust_ferrari_canny']       #Y of our model
print(grasp_metrics.shape)
print(np.max(grasp_metrics))