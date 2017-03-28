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
'''
from orthaffine import OrthAffine as OA
theta = 30.0/180.0*3.14
oa = OA(theta)
oa.readpcd('box_points.pcd')
oa.affine_pro()
'''
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
		self.box_size = rospy.get_param("box_size")
		self.bridge = CvBridge()
		rospy.loginfo("Orthaffine Initialized!")

	def readpcd(self,filename):
		if filename.split('.')[1] != 'pcd':
			return False
		else:
			self.p = pcl.load(filename)
			self.points = self.p.to_array()
			return True

	def readpoints(self,points):
		self.points = points

	def affine(self):
		yz = self.points[:,1:3]
		new_y = np.dot(yz,np.transpose(self.R))[:,0]
		self.points[:,1] = new_y

	def affine_pro(self):
		step_size = (self.bmX - self.bnX)/self.image_size
		nm = int((self.box_size - self.image_size)/2.0)
		lf_x = np.asarray([self.bnX-step_size*i for i in range(nm)]).reshape((-1,1))
		lf_x = np.repeat(lf_x,self.image_size,axis=1).reshape(-1,1)
		lf_y = np.asarray([self.bmY-step_size*i for i in range(self.image_size)]).reshape((-1,1))
		lf_y = np.repeat(lf_y,nm,axis=1).transpose().reshape(-1,1)
		lf_xy = np.concatenate((lf_x,lf_y),axis=1)
		
		rt_x = np.asarray([self.bmX+step_size*i for i in range(nm)]).reshape((-1,1))
		rt_x = np.repeat(rt_x,self.image_size,axis=1).reshape(-1,1)
		rt_y = np.asarray([self.bmY-step_size*i for i in range(self.image_size)]).reshape((-1,1))
		rt_y = np.repeat(rt_y,nm,axis=1).transpose().reshape(-1,1)
		rt_xy = np.concatenate((rt_x,rt_y),axis=1)
		
		up_x = np.asarray([self.bnX+step_size*i for i in range(self.image_size)]).reshape((-1,1))
		up_x = np.repeat(up_x,nm,axis=1).reshape(-1,1)
		up_y = np.asarray([self.bmY+step_size*i for i in range(nm)]).reshape((-1,1))
		up_y = np.repeat(up_y,self.image_size,axis=1).transpose().reshape(-1,1)
		up_xy = np.concatenate((up_x,up_y),axis=1)
		
		lw_x = np.asarray([self.bnX+step_size*i for i in range(self.image_size)]).reshape((-1,1))
		lw_x = np.repeat(lw_x,nm,axis=1).reshape(-1,1)
		lw_y = np.asarray([self.bnY-step_size*i for i in range(nm)]).reshape((-1,1))
		lw_y = np.repeat(lw_y,self.image_size,axis=1).transpose().reshape(-1,1)
		lw_xy = np.concatenate((lw_x,lw_y),axis=1)
		
		ex_xy = np.concatenate((lf_xy,rt_xy,up_xy,lw_xy),axis=0).astype(np.float32)

		ex_z = griddata(self.points[:,0:2],self.points[:,2],(ex_xy[:,0],ex_xy[:,1]),method='nearest',fill_value=0.0)
		ex_pts = np.concatenate((ex_xy,ex_z.reshape(-1,1)),axis=1)
		self.points = np.concatenate((self.points,ex_pts),axis=0)

		yz = self.points[:,1:3]
		new_y = np.dot(yz,np.transpose(self.R))[:,0]
		self.points[:,1] = new_y


	def interpolate_large(self,theta):
		'''This method is only used to interpolate large box'''
		xy = self.points[:,0:2]
		z = self.points[:,2]
		x_step_size = (self.bmX - self.bnX)/self.image_size
		y_step_size = (self.bmY - self.bnY)/self.image_size*cos(theta)
		delta = (self.box_size - self.image_size)/2
		x_start = self.bnX - x_step_size*delta
		y_start = self.bnY - y_step_size*delta
		grid_x = np.asarray([x_start+x_step_size*i for i in range(self.box_size)]).reshape((-1,1))
		grid_x = np.repeat(grid_x,self.box_size,axis=1)
		grid_y = np.asarray([y_start*cos(theta)+y_step_size*i for i in range(self.box_size)]).reshape((-1,1))
		grid_y = np.repeat(grid_y,self.box_size,axis=1).transpose()
		grid_z = griddata(xy,z,(grid_x,grid_y),method='nearest',fill_value=0.0)
		new_points = np.asarray([grid_x,grid_y,grid_z]).transpose().reshape((-1,3)).astype(np.float32)
		#print(new_points.transpose().shape)
		#new_points = np.swapaxes(new_points,0,2).reshape((-1,3)).astype(np.float32)
		#print(new_points.shape)
		self.points = new_points.copy()

	def interpolate_small(self,theta):
		'''This method is only used to interpolate small box defined by its arguments'''
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

	def interpolate(self,theta):
		'''This method is only used to interpolate small box defined by its arguments'''
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
		'''for both small and large depending on the previous'''
		self.p.from_array(self.points)
		self.p._to_pcd_file(filename)

	def project_large(self):
		'''only for large box'''
		z = self.points[:,2]
		z_mean = (self.bmZ - self.bnZ)/2.0
		z = (z - z_mean)/(self.bmZ - self.bnZ)
		self.image_numpy = np.flipud(z.reshape(self.box_size,self.box_size))
		return z

	def project_small(self):
		'''only for small box'''
		z = self.points[:,2]
		z_mean = (self.bmZ - self.bnZ)/2.0
		z = (z - z_mean)/(self.bmZ - self.bnZ)
		self.image_numpy = np.flipud(z.reshape(self.image_size,self.image_size))
		return z
		
	def saveimage(self,filename):
		'''saveimage can save small box image or large box image depending on what interpolate and project methods are used previously'''
		z_mean = (self.bmZ - self.bnZ)/2.0
		vmin = (self.bnZ - z_mean)/(self.bmZ - self.bnZ)
		vmax = (self.bmZ - z_mean)/(self.bmZ - self.bnZ)
		plt.imshow(self.image_numpy,cmap='Greys_r', vmin=vmin, vmax=vmax)
		plt.savefig(filename)

	def publishimage(self):
		'''for both small and large depending on the previous'''
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
		image_array = self.project_small()
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



