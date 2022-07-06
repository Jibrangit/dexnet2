# DEX-NET 2.0 DATASET

# OVERVIEW
This is the latest version of the dataset used to train the Grasp-Quality-Convolutional-Neural-Network from Dex-Net 2.0.
The dataset contains approximately 6.7 million datapoints generated from 1,500 3D object models from the KIT and 3DNet datasets.
The dataset was generated on April 9, 2017.
For more information on the dataset generation method, please see our paper at https://arxiv.org/abs/1703.09312.

If you use this dataset in a publication, please cite:

Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea,
and Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic
Grasp Metrics." Robotics, Science, and Systems, 2017. Cambridge, MA.

# DATA EXTRACTION
tar -xvzf dexnet_2.tar.gz

# DATA FORMAT
Files are in compressed numpy (.npz) format and organized by attribute.
Each file contains 1,000 datapoints except for the last file (6728), which contains 850 datapoints.
There are five different attributes:

  1) depth_ims_tf_table:
       Description: depth images transformed to align the grasp center with the image center and the grasp axis with the middle row of pixels
       File dimension: 1000x32x32x1 (except the last file)
       Organization: {num_datapoints} x {image_height} x {image_width} x {num_channels}
       Notes: Rendered with OSMesa using the parameters of a Primesense Carmine 1.08

  2) hand_poses:
       Description: configuration of the robot gripper corresponding to the grasp
       File dimension: 1000x7 (except the last file)
       Organization: {num_datapoints} x {hand_configuration}, where columns are
         0: row index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
         1: column index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
         2: depth, in meters, of gripper center from the camera that took the corresponding depth image
	 3: angle, in radians, of the grasp axis from the image x-axis (middle row of pixels, pointing right in image space)
	 4: row index, in pixels, of the object center projected into a depth image centered on the world origin
	 5: column index, in pixels, of the object center projected into a depth image centered on the world origin
	 6: width, in pixels, of the gripper projected into the depth image
       Notes: To replicate the Dex-Net 2.0 results, you only need column 2.
         The gripper width was 5cm, corresponding to the width of our custom ABB YuMi grippers

  3) robust_ferrari_canny:
       Description: value of the robust epsilon metric computed according to the Dex-Net 2.0 graphical model
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Threshold the values in this value by 0.002 to generate the 0-1 labels of Dex-Net 2.0

  4) ferrari_canny:
       Description: value of the epsilon metric, without measuring robustness to perturbations in object pose, gripper pose, and friction
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Not used in the Dex-Net 2.0 paper. Included for comparison purposes

  5) force_closure:
       Description: value of force closure, without measuring robustness to perturbations in object pose, gripper pose, and friction
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Not used in the Dex-net 2.0 paper. Included for comparison purposes

# DATA INDEXING
The index into the first dimension of each file corresponds to a different attribute.
Furthermore, the same index in the file with the same number corresponds to the same datapoints.

For example, datapoint 2478 could be accessed in Python as:
  depth_im = np.load('depth_ims_tf_table_00002.npz')['arr_0'][478,...]
  hand_pose = np.load(hand_poses_00002.npz')['arr_0'][478,...]
  grasp_metric = np.load(robust_ferrari_canny_00002.npz')['arr_0'][478,...]
