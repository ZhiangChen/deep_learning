#!/usr/bin/env python

import pcl
import numpy as np
from math import *

theta = 20.0/180.0*pi
p = pcl.load('box_points.pcd')
points = p.to_array()
#x = points[:,0]
#y = points[:,1]
#z = points[:,2]
yz = points[:,1:3]
R = np.asarray([[cos(-theta),-sin(-theta)],[sin(-theta),cos(-theta)]])
new_yz = np.dot(yz,np.transpose(R))
new_y = new_yz[:,0]
points[:,1] = new_y
p.from_array(points)
p._to_pcd_file('affined_box_points.pcd')

