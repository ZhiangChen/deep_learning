#!/usr/bin/env python2

# MIT License
#
# Copyright (c) 2016 Zhiang Chen

'''
Receive the cropped image from "cropped_depth_image", and shift it and save the shifted images.
'''
from __future__ import print_function
import rospy
import roslib
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import time
import os
import sys

image_size = 34
num_shift = 3 #3x3
lf_x = rospy.get_param("lf_x")
lf_y = rospy.get_param("lf_y")
rt_x = lf_x + image_size
rt_y = lf_y + image_size

orientation = 0
face = 0


class savor:
	def __init__(self):
		'''Initialize ros publisher, subscriber'''
		self.sub = rospy.Subscriber('depth_image',Image,self.callback,queue_size=1)
		self.pub = rospy.Publisher('/cropped_depth_image',Image,queue_size=1)
		self.bridge = CvBridge()
		rospy.loginfo("Initialized!")

	def callback(self,data):
		cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
		cropped_image = cv_image[lf_y: rt_y, lf_x : rt_x].reshape((image_size, image_size))
		ros_image = self.bridge.cv2_to_imgmsg(cropped_image, encoding="mono8")
		self.pub.publish(ros_image)
		name = raw_input("Input the name of object (or Enter to update): ")
		if name != '':
			position = raw_input("Input the position of the object: ")
			for i in range(num_shift):
				for j in range(num_shift):
					index = 3*j+i
					# crop image
					cropped_image = cv_image[lf_y+j : rt_y+j, lf_x+i : rt_x+i].reshape((image_size, image_size))
					cropped_name = wd+"/cropped_"+name+'_p'+str(position)+'_f'+str(face)+'_r'+str(orientation)+'_'+str(index)+'.bmp'
					cv2.imwrite(cropped_name,cropped_image)
		

		          	

if __name__ == '__main__':
	rospy.init_node('savor',anonymous=True)
	wd = os.getcwd()
	print("Current directory is \""+wd+"\"")
	cmd = raw_input("Start to save the images in this directory? (yes/no) ")
	assert cmd == "yes" or cmd == "no"
	if cmd == "no":
		print("Input correct directory:")
		wd = raw_input()
	assert os.path.isdir(wd)
	ev = savor()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node evaluate_image")
  		