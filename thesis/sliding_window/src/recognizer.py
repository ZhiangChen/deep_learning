#! /usr/bin/python
'''
Recognizer
Zhiang Chen, April 2017

The MIT License (MIT)
Copyright (c) 2017 Zhiang Chen
'''


import rospy
import roslib
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from orthaffine import OrthAffine as OA
from evaluator import *
from math import *

theta = 30.0/180*pi

class Recognizer():
	def __init__(self):
		self.sub = rospy.Subscriber('/box_points', PointCloud2, self.callback, queue_size=1)
		self.pub = 
		self.theta = theta
		self.oa = OA(theta)
		self.ev = evaluator()
