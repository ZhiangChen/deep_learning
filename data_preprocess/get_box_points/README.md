# get_box_points

### 1. ROS Node: get_box_points
subscribing to kinect2/qhd/points, transforming the point clouds to plane frame, getting points in box, and saving the points in file 'box_points.pcd'

### 2. reassign_points.py
Loading points from 'box_points.pcd', interpolating points on a grids, saving the new points. 

### [*python-pcl*](https://github.com/strawlab/python-pcl)
Instruction for installing python-pcl:
```shell
sudo install Cython
sudo python setup.py clean
sudo make clean
sudo make all
sudo python setup.py install
```
