#!/usr/bin/env python

import pcl
import numpy as np
from scipy.interpolate import griddata

file_name = 'affined_box_points.pcd'
p = pcl.load(file_name)
points = p.to_array()
xy = points[:,0:2]
z = points[:,2]
grid_x = np.asarray([-0.09+0.0018*i for i in range(101)]).reshape((-1,1))
grid_x = np.repeat(grid_x,101,axis=1)
grid_y = np.transpose(grid_x)
grid_z = griddata(xy,z,(grid_x,grid_y),method='nearest')
new_points = np.asarray([grid_x,grid_y,grid_z])
new_points = np.swapaxes(new_points,0,2).reshape((-1,3)).astype(np.float32)
p.from_array(new_points)
p._to_pcd_file('my_box_points.pcd')

'''
rosrun rviz rviz
rosrun pcl_utils display_pcd_file
file name = my_box_points.pcd
frame = camera_depth_optical_frame

'''

