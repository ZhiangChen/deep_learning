`
cd ~/catkin_ws/src/deep_learning/thesis/supervised_learning
roslaunch kinect2_bridge kinect2_bridge.launch
rosrun rviz rviz
rosparam load auto_recognition2.yaml
rosrun depthnet box_points_pub
rosrun orthaffine_projection orthaffine.py
python evaluator.py
rostopic echo prediction
rviz: add image by topic 'p_box_image/image'
`
