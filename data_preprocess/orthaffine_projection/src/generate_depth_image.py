#!/usr/bin/env python

import pcl
import cv2
import numpy as np
import matplotlib.pyplot as plt

file_name = 'my_box_points.pcd'
p = pcl.load(file_name)
points = p.to_array()
z = points[:,2]
z_mean = np.mean(z)
z = z - z_mean
z_max = np.amax(z)
z_min = np.amin(z)
z = z/(z_max-z_min)
image = z.reshape((101,101))
plt.imshow(image,cmap='Greys_r')
plt.show()

