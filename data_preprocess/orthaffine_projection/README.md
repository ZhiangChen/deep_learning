# Orthaffine Projection

### 1. get_box_points (ROS node)
subscribing to kinect2/qhd/points, transforming the point clouds to plane frame, getting points in box, and saving the points in file 'box_points.pcd'

### 2. get_box_points_from_files (ROS node)
Reading the names of the pcd files in name_list (data_pp/list_names.py), loading the parameters of box filter, getting the points in the box, and saving the points in file as name rule: 'box_phase_objectname_orientation.pcd' 

### 3. interpolate.py
Loading points from 'affined_box_points.pcd', interpolating points on a grids, saving the new points in file 'my_box_points.pcd'

### 4. select_area_from_file (ROS node)
Providing a method of getting the parameters of box filter by reading a pcd file.

### 5. orthaffine.py (ROS node)
Subscribing to 'box_points', converting points cloud with a affine transformation, interpolating the affined point cloud, projecting the affined point cloud with a orthographic projection; publishing the depth information by numpy_msg(Floats) on topic '/box_image/numpy' and depth image by sensor_msgs/Image on topic '/box_image/image'.

OrthAffine also provides functions to operation on pcd files.
* OrthAffine.readpcd(filename): read a pcd file by filename
* OrthAffine.affine(): have a affine transformation
* OrthAffine.interpolation(theta): interpolate the point cloud, theta is the angle of the rotation angle along x axis.
* OrthAffine.savepcd(filename): save the point cloud as filename
* OrthAffine.saveimage(filename): save the depth image as filename
* OrthAffine.project(): project the point cloud on x-y plane with orthographic projection

### 6. cropper.py (ROS node)
Subscribing to the topic '/box_image/numpy'; croping the a 80x80 image from the center of the image; publishing to '/cropped_box_image/numpy'.

cropper.SaveRandomCropped also provides functions to randomly crop the depth images that are in cPickle format 'depth_data_numpy' and save the cropped images as 'cropped_depth_data_numpy'.

### [*python-pcl*](https://github.com/strawlab/python-pcl)
Instruction for installing python-pcl:
```shell
sudo install Cython
sudo python setup.py clean
sudo make clean
sudo make all
sudo python setup.py install
```
