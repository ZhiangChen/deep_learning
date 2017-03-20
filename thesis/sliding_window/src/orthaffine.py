#!/usr/bin/env python
'''orthaffine'''

import pcl
import numpy as np
from math import *
import rospy 
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from scipy.interpolate import griddata
import sys
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

class OrthAffine():
	def __init__(self,theta):
		'unit of theta is rad'
		self.theta = theta
		self.R = np.asarray([[cos(-theta),-sin(-theta)],[sin(-theta),cos(-theta)]])
		self.sub = rospy.Subscriber('/box_points', PointCloud2, self.callback, queue_size=1)
		self.pub1 = rospy.Publisher('/box_image/numpy', numpy_msg(Floats), queue_size=1)
		self.pub2 = rospy.Publisher('/box_image/image', Image, queue_size=1)
		self.bnX = rospy.get_param("bnX")
		self.bnY = rospy.get_param("bnY")
		self.bnZ = rospy.get_param("bnZ")
		self.bmX = rospy.get_param("bmX")
		self.bmY = rospy.get_param("bmY")
		self.bmZ = rospy.get_param("bmZ")
		self.image_size = rospy.get_param("image_size")
		self.bridge = CvBridge()
		rospy.loginfo("Initialized!")

	def readpcd(self,filename):
		if filename.split('.')[1] != 'pcd':
			return False
		else:
			self.p = pcl.load(filename)
			self.points = self.p.to_array()
			return True

	def readpoints(self,points):
		self.points = points.copy()

	def affine(self):
		yz = self.points[:,1:3]
		new_y = np.dot(yz,np.transpose(self.R))[:,0]
		self.points[:,1] = new_y

	def interpolate(self,theta):
		xy = self.points[:,0:2]
		z = self.points[:,2]
		x_step_size = (self.bmX - self.bnX)/self.image_size
		y_step_size = (self.bmY - self.bnY)/self.image_size*cos(theta)
		grid_x = np.asarray([self.bnX+x_step_size*i for i in range(self.image_size)]).reshape((-1,1))
		grid_x = np.repeat(grid_x,self.image_size,axis=1)
		grid_y = np.asarray([self.bnY*cos(theta)+y_step_size*i for i in range(self.image_size)]).reshape((-1,1))
		grid_y = np.repeat(grid_y,self.image_size,axis=1).transpose()
		grid_z = griddata(xy,z,(grid_x,grid_y),method='nearest',fill_value=0.0)
		new_points = np.asarray([grid_x,grid_y,grid_z]).transpose().reshape((-1,3)).astype(np.float32)
		#print(new_points.transpose().shape)
		#new_points = np.swapaxes(new_points,0,2).reshape((-1,3)).astype(np.float32)
		#print(new_points.shape)
		self.points = new_points.copy()

	def savepcd(self,filename):
		self.p.from_array(self.points)
		self.p._to_pcd_file(filename)

	def project(self):
		z = self.points[:,2]
		z_mean = (self.bmZ - self.bnZ)/2.0
		z = (z - z_mean)/(self.bmZ - self.bnZ)
		self.image_numpy = np.flipud(z.reshape(self.image_size,self.image_size))
		return z
		
	def saveimage(self,filename):
		z_mean = (self.bmZ - self.bnZ)/2.0
		vmin = (self.bnZ - z_mean)/(self.bmZ - self.bnZ)
		vmax = (self.bmZ - z_mean)/(self.bmZ - self.bnZ)
		plt.imshow(self.image_numpy,cmap='Greys_r', vmin=vmin, vmax=vmax)
		plt.savefig(filename)

	def publishimage(self):
		image = ((self.image_numpy + 0.65)*255).astype(np.uint8)
		ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
		self.pub2.publish(ros_image)

	def callback(self,box_points):
		generator = pc2.read_points(box_points, skip_nans=True, field_names=("x", "y", "z"))
		pts = list()
		for i in generator:
			pts.append(i)
		self.points = np.asarray(pts)
		self.affine()
		self.interpolate(self.theta)
		image_array = self.project()
		self.pub1.publish(image_array)
		self.publishimage()
		sys.stdout.write(".")
		sys.stdout.flush()

if __name__=='__main__':
	rospy.init_node('OrthAffine',anonymous=True)
	theta = 30.0/180.0*pi
	orthaffine = OrthAffine(theta)
	#orthaffine.readpcd('box_points.pcd')
	#orthaffine.savepcd('test_box.pcd')
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node OrthAffine")



