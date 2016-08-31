# [auto recognition](https://www.youtube.com/watch?v=RnQiQT0xHU0)
---
## Description
This is a ros package for 3D objects recognition. 

## Dependencies
- ROS Indigo
- Kinect2 Bridge
- Tensorflow (Python2)
- [pcl_utils](https://github.com/wsnewman/learning_ros/tree/master/Part_3/pcl_utils)

## ROS Nodes
#### 1. selectArea (cpp)
Require users select an area using "PublishSelectedPoints" in RViz. Then store the parameters in a yaml file.

#### 2. kinect2depth (cpp)
Receive the pointcloud data from "kinect2/qhd/points", and then project the points in the box filter to a depth image. Publish the depth image to "depth_image" with 10hz.

#### 3. depth2input (python2)
Receive the depth image from "depth_image", and crop the image by a 34x34 window. Then publish the cropped image to "cropped_depth_image" with 10hz.

#### 4. evaluate_image (python2)
Receive the cropped image from "cropped_depth_image", and reformat its shape to be compatible with the input shape for the network. And then evaluate the input.

## Usage
```shell
roslaunch kinect2_bridge kinect2_bridge.launch
rosrun auto_recognition2 selectArea
rosparam load auto_recognition2.yaml
rosrun auto_recognition2 kinect2depth2
rosrun auto_recognition2 depth2input.py
rosrun auto_recognition2 evaluate_image.py
rosrun image_view image_view image:=cropped_depth_image
rostopic echo prediction
```

## Topics
- /kinect2/qhd/points
- /depth_image
- /cropped_depth_image
- /prediction
