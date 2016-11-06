# DepthNet
---
Zhiang Chen, Nov 2016

### DepthNet
![alt tag](./DepthNet.png)

```shell
roslaunch kinect2_bridge kinect2_bridge.launch
rosparam load ...
rosrun depthnet box_points_pub
rosrun orthaffine_projection orthaffine.py
rosrun orthaffine_projection cropper.py
rosrun depthnet evaluator.py
rostopic echo prediction
rosrun image_view image_view image:=/cropped_box_image/image
```
