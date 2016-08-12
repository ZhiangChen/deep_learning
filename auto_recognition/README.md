# auto recognition
---
### 1. kinect2depth (cpp)
Receive the pointcloud data from "kinect2/qhd/points", and then project the points in the box filter to a depth image. Publish the depth image to "depth_image" with 10hz.

### 2. depth2input (python2)
Receive the depth image from "depth_image", and crop the image by a 34x34 window. Then publish the cropped image to "cropped_depth_image" with 10hz.

### 3. evaluate_image (python2)
Receive the cropped image from "cropped_depth_image", and reformat its shape to be compatible with the input shape for the network. And then evaluate the input.
